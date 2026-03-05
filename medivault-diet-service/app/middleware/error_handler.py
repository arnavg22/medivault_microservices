"""
Global exception handlers for the FastAPI application.

Registers handlers for:
  - AllProvidersExhaustedException → 502
  - RequestValidationError → 422
  - ValueError → 400
  - Generic Exception → 500
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.services.llm.base import AllProvidersExhaustedException

logger = structlog.get_logger("middleware.error_handler")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all custom exception handlers to the app instance."""

    @app.exception_handler(AllProvidersExhaustedException)
    async def all_providers_exhausted_handler(
        request: Request, exc: AllProvidersExhaustedException
    ) -> JSONResponse:
        logger.error(
            "all_llm_providers_exhausted",
            path=request.url.path,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "error": "All AI providers are currently unavailable",
                "detail": str(exc),
                "code": "ALL_PROVIDERS_EXHAUSTED",
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "request_validation_error",
            path=request.url.path,
            errors=exc.errors(),
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "detail": exc.errors(),
                "code": "VALIDATION_ERROR",
            },
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        logger.warning(
            "value_error",
            path=request.url.path,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Bad request",
                "detail": str(exc),
                "code": "BAD_REQUEST",
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred. Please try again later.",
                "code": "INTERNAL_ERROR",
            },
        )
