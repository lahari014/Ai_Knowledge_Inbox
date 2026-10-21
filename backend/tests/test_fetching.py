import httpx
import pytest
from app.api.errors import AppError
from app.core.config import Settings
from app.services.fetching import ContentFetcher, extract_readable_html, is_public_ip

PUBLIC = "1.1.1.1"


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.8",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
        "240.0.0.1",
        "::1",
        "fc00::1",
    ],
)
def test_non_public_addresses_are_rejected(address: str) -> None:
    assert is_public_ip(address) is False


def test_public_address_is_allowed() -> None:
    assert is_public_ip(PUBLIC) is True


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://public.test/file",
        "http://user:pass@public.test/a",
        "http://localhost/admin",
    ],
)
def test_unsafe_urls_are_rejected_before_a_request(url: str) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(200, text="nope")

    fetcher = _fetcher(handler)
    with pytest.raises(AppError) as caught:
        fetcher.fetch(url)

    assert caught.value.status_code == 422
    assert caught.value.code == "UNSAFE_URL"
    assert calls["count"] == 0


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.1.2.3", "192.168.1.8", "169.254.1.1", "224.0.0.1", "0.0.0.0", "240.0.0.1"],
)
def test_resolved_non_public_destination_is_not_requested(address: str) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        calls["count"] += 1
        return httpx.Response(200, text="secret", headers={"content-type": "text/plain"})

    fetcher = _fetcher(handler, resolver=lambda hostname: [address])
    with pytest.raises(AppError) as caught:
        fetcher.fetch("http://public.test/secret")

    assert caught.value.code == "UNSAFE_URL"
    assert calls["count"] == 0


def test_redirect_to_private_address_is_revalidated() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://private.test/hidden"})

    def resolver(hostname: str) -> list[str]:
        return {"public.test": [PUBLIC], "private.test": ["127.0.0.1"]}[hostname]

    fetcher = _fetcher(handler, resolver=resolver)
    with pytest.raises(AppError) as caught:
        fetcher.fetch("http://public.test/start")

    assert caught.value.code == "UNSAFE_URL"
    assert calls == ["http://public.test/start"]


def test_three_redirects_can_reach_a_public_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        number = int(request.url.path.strip("/") or "0")
        if number >= 3:
            return httpx.Response(200, text="Final page", headers={"content-type": "text/plain"})
        return httpx.Response(302, headers={"location": f"http://public.test/{number + 1}"})

    fetched = _fetcher(handler).fetch("http://public.test/0")

    assert fetched.text == "Final page"
    assert fetched.url == "http://public.test/3"


def test_fourth_redirect_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        number = int(request.url.path.strip("/") or "0")
        return httpx.Response(302, headers={"location": f"http://public.test/{number + 1}"})

    with pytest.raises(AppError) as caught:
        _fetcher(handler).fetch("http://public.test/0")

    assert caught.value.status_code == 422
    assert "redirected" in caught.value.message


def test_response_over_byte_limit_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, content=b"1234567890", headers={"content-type": "text/plain"})

    settings = _settings().model_copy(update={"fetch_max_bytes": 4})
    fetcher = ContentFetcher(settings, client=_client(handler), resolver=lambda hostname: [PUBLIC])
    with pytest.raises(AppError) as caught:
        fetcher.fetch("http://public.test/big")

    assert caught.value.code == "RESPONSE_TOO_LARGE"


def test_unsupported_content_type_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"hello": "world"})

    with pytest.raises(AppError) as caught:
        _fetcher(handler).fetch("http://public.test/data")

    assert caught.value.code == "UNSUPPORTED_CONTENT"


def test_timeout_maps_to_gateway_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        raise httpx.ReadTimeout("slow")

    with pytest.raises(AppError) as caught:
        _fetcher(handler).fetch("http://public.test/slow")

    assert caught.value.status_code == 504
    assert caught.value.code == "UPSTREAM_TIMEOUT"


def test_html_extraction_keeps_readable_text_only() -> None:
    html = """
    <html><head><title>Docs Title</title><style>p{}</style></head>
    <body><nav>Skip</nav><script>alert(1)</script><main><p>Useful knowledge.</p></main>
    <footer>Footer</footer></body></html>
    """

    title, text = extract_readable_html(html, "fallback")

    assert title == "Docs Title"
    assert "Useful knowledge." in text
    assert "alert" not in text
    assert "Skip" not in text
    assert "Footer" not in text


def _settings() -> Settings:
    return Settings(openai_api_key="", _env_file=None)  # type: ignore[call-arg]


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)  # type: ignore[arg-type]


def _fetcher(handler: object, resolver: object | None = None) -> ContentFetcher:
    return ContentFetcher(
        _settings(),
        client=_client(handler),
        resolver=resolver or (lambda hostname: [PUBLIC]),  # type: ignore[arg-type]
    )
