"""
Request ID Middleware — attaches a unique X-Request-ID to every request/response.

If the incoming request already carries an X-Request-ID header (e.g. from an
API gateway), it is preserved. Otherwise a short 8-character UUID is generated.
The ID is also bound to structlog's context variables so every log line within
the request lifecycle includes it automatically.
"""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every HTTP exchange."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_HEADER) or uuid.uuid4().hex[:8]

        # Bind to structlog context so all downstream log calls include it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        response: Response = await call_next(request)
        response.headers[_HEADER] = request_id
        return response
