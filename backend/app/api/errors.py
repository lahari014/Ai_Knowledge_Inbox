from dataclasses import dataclass

from starlette.responses import JSONResponse


@dataclass
class AppError(Exception):
    code: str
    message: str
    status_code: int

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)


def error_payload(code: str, message: str, request_id: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message, "requestId": request_id}}


def error_response(code: str, message: str, status_code: int, request_id: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=error_payload(code, message, request_id))
