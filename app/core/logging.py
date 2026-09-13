import logging
import sys

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure concise structured logs for the application process."""
    level = logging.DEBUG if settings.debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )