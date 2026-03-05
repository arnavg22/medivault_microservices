"""Entry point for the medivault-diet-service."""

import uvicorn

from app.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.node_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
