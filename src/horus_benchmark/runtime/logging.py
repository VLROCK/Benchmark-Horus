"""Central logging configuration for local and cluster execution."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    *,
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure consistent console logging and an optional UTF-8 file."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        output_path = Path(log_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=force,
    )
    return logging.getLogger("horus_benchmark")

