import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import ParseResult, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.api.errors import AppError
from app.core.config import Settings
from app.domain.models import FetchedContent

BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "metadata.google.internal"}
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class ContentFetcher:
    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
        resolver: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(settings.fetch_timeout_seconds),
            follow_redirects=False,
        )
        self._resolver = resolver or resolve_public_hostname

    def fetch(self, url: str) -> FetchedContent:
        current = url
        for redirect_index in range(self._settings.fetch_max_redirects + 1):
            self._validate_url(current)
            try:
                with self._client.stream("GET", current) as response:
                    if response.status_code in REDIRECT_STATUSES:
                        if redirect_index == self._settings.fetch_max_redirects:
                            raise AppError("UNSAFE_URL", "The page redirected too many times.", 422)
                        location = response.headers.get("location")
                        if not location:
                            raise AppError(
                                "FETCH_FAILED",
                                "The page returned a redirect without a destination.",
                                502,
                            )
                        current = urljoin(current, location)
                        continue
                    if response.status_code >= 400:
                        raise AppError("FETCH_FAILED", "The page could not be fetched.", 502)
                    content_type = (
                        response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    )
                    if content_type not in {"text/html", "text/plain"}:
                        raise AppError(
                            "UNSUPPORTED_CONTENT",
                            "Only HTML and plain-text pages can be saved.",
                            422,
                        )
                    body = self._read_limited(response)
            except AppError:
                raise
            except httpx.TimeoutException as exc:
                raise AppError("UPSTREAM_TIMEOUT", "Fetching the page timed out.", 504) from exc
            except httpx.HTTPError as exc:
                raise AppError("FETCH_FAILED", "The page could not be fetched.", 502) from exc
            return self._to_content(current, content_type, body)
        raise AppError("UNSAFE_URL", "The page redirected too many times.", 422)

    def _to_content(self, url: str, content_type: str, body: bytes) -> FetchedContent:
        parsed = urlparse(url)
        fallback_title = parsed.hostname or url
        text_body = body.decode("utf-8", errors="replace")
        if content_type == "text/plain":
            title = fallback_title
            text = normalize_text(text_body)
        else:
            title, text = extract_readable_html(text_body, fallback_title)
        if not text.strip():
            raise AppError("EMPTY_CONTENT", "The page did not contain readable text.", 422)
        return FetchedContent(url=url, title=title[:300], text=text)

    def _validate_url(self, url: str) -> ParseResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise AppError("UNSAFE_URL", "Only public HTTP and HTTPS URLs can be saved.", 422)
        if parsed.username or parsed.password:
            raise AppError("UNSAFE_URL", "URLs with embedded credentials cannot be saved.", 422)
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".local"):
            raise AppError("UNSAFE_URL", "That address is not a public web page.", 422)
        try:
            addresses = self._resolver(hostname)
        except socket.gaierror as exc:
            raise AppError("FETCH_FAILED", "The page address could not be resolved.", 502) from exc
        if not addresses or any(not is_public_ip(address) for address in addresses):
            raise AppError("UNSAFE_URL", "That address is not a public web page.", 422)
        return parsed

    def _read_limited(self, response: httpx.Response) -> bytes:
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._settings.fetch_max_bytes:
            raise AppError(
                "RESPONSE_TOO_LARGE", "The page is larger than the configured limit.", 422
            )
        body = bytearray()
        for chunk in response.iter_bytes():
            if len(body) + len(chunk) > self._settings.fetch_max_bytes:
                raise AppError(
                    "RESPONSE_TOO_LARGE", "The page is larger than the configured limit.", 422
                )
            body.extend(chunk)
        return bytes(body)


def resolve_public_hostname(hostname: str) -> list[str]:
    infos = socket.getaddrinfo(hostname, None)
    return [str(info[4][0]) for info in infos]


def is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or not getattr(ip, "is_global", False)
    )


def extract_readable_html(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else fallback_title
    text = normalize_text(soup.get_text("\n", strip=True))
    return title or fallback_title, text


def normalize_text(value: str) -> str:
    lines = [" ".join(line.split()) for line in value.splitlines()]
    return "\n".join(line for line in lines if line).strip()
