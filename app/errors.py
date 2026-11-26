from __future__ import annotations

from enum import StrEnum


class ErrorKind(StrEnum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONTENT_FILTER = "content_filter"
    PROVIDER = "http_5xx"
    AUTH_BILLING = "auth_billing"
    BAD_REQUEST = "bad_request"
    BUDGET = "budget_exhausted"
    DEADLINE = "deadline_exceeded"
    PARSE = "parse_error"
    CALLBACK = "callback_error"
    UNKNOWN = "unknown"


class ReviewError(Exception):
    """可归类的模型调用错误，供重试与降级策略使用。"""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        model: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.model = model
        self.retry_after = retry_after
        self.detail = message

    @property
    def retryable(self) -> bool:
        return self.kind in {
            ErrorKind.TIMEOUT,
            ErrorKind.RATE_LIMIT,
            ErrorKind.PROVIDER,
        }

    @property
    def fatal(self) -> bool:
        return self.kind == ErrorKind.AUTH_BILLING
