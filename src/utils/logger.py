"""
SiliconPilot AI — Logging Utility
==================================
get_logger(name) returns a named logger configured with:
  • A coloured console handler  (uses *colorlog* when available, falls back
    to stdlib's StreamHandler with a plain formatter)
  • A RotatingFileHandler writing to LOG_FILE (5 MB per file, 3 backups)

The log directory is created automatically if it does not exist.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Import settings — graceful fallback so the logger works stand-alone
# ---------------------------------------------------------------------------
try:
    from config.settings import LOG_FILE, LOG_DIR
except ImportError:
    # Fallback: write next to this file's project root
    _here = Path(__file__).resolve()
    LOG_DIR = _here.parents[3] / "logs"
    LOG_FILE = LOG_DIR / "siliconpilot.log"

# ---------------------------------------------------------------------------
# Colourlog — optional dependency
# ---------------------------------------------------------------------------
try:
    import colorlog  # type: ignore
    _COLORLOG_AVAILABLE = True
except ImportError:
    _COLORLOG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_BYTES: int = 5 * 1024 * 1024   # 5 MB
_BACKUP_COUNT: int = 3
_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FMT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

_COLOR_MAP: dict[str, str] = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "bold_red",
}

# Keep track of loggers we have already configured so we never add duplicate
# handlers when get_logger() is called more than once with the same name.
_configured: set[str] = set()


def _ensure_log_dir() -> None:
    """Create the log directory (and parents) if it does not already exist."""
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


def _make_console_handler() -> logging.Handler:
    """Return a coloured (or plain) console handler writing to *stderr*."""
    if _COLORLOG_AVAILABLE:
        handler = colorlog.StreamHandler(sys.stderr)
        handler.setFormatter(
            colorlog.ColoredFormatter(
                fmt="%(log_color)s" + _CONSOLE_FMT,
                datefmt=_DATE_FMT,
                log_colors=_COLOR_MAP,
                reset=True,
                style="%",
            )
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(fmt=_CONSOLE_FMT, datefmt=_DATE_FMT)
        )
    handler.setLevel(logging.DEBUG)
    return handler


def _make_file_handler() -> logging.Handler:
    """Return a rotating file handler targeting LOG_FILE."""
    _ensure_log_dir()
    handler = RotatingFileHandler(
        filename=str(LOG_FILE),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(fmt=_FILE_FMT, datefmt=_DATE_FMT)
    )
    handler.setLevel(logging.DEBUG)
    return handler


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a configured :class:`logging.Logger` for *name*.

    Parameters
    ----------
    name:
        The logger name, typically ``__name__`` of the calling module.
    level:
        Root level for this logger (default ``DEBUG`` — handlers may filter
        further).

    Returns
    -------
    logging.Logger
        A logger with a coloured console handler and a rotating file handler
        already attached.  Calling this function a second time with the same
        *name* returns the existing logger unchanged.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    logger.setLevel(level)
    logger.addHandler(_make_console_handler())
    logger.addHandler(_make_file_handler())
    # Prevent log records from bubbling up to the root logger (avoids
    # duplicate output when the root logger also has handlers).
    logger.propagate = False

    _configured.add(name)
    return logger


# ---------------------------------------------------------------------------
# Module-level convenience logger (used when this module is run directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _log = get_logger("siliconpilot.logger_test")
    _log.debug("DEBUG message — logger is working.")
    _log.info("INFO  message — SiliconPilot AI logger initialised.")
    _log.warning("WARNING message — check your configuration.")
    _log.error("ERROR message — simulated error (not a real error).")
    _log.critical("CRITICAL message — simulated critical (not real).")
