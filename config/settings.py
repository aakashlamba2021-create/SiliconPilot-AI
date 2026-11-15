"""
SiliconPilot AI — Central Configuration
All paths are derived relative to this file; no hardcoded user paths.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from config/settings.py → SiliconPilot-AI/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "samples"

# Dataset-specific sub-directories
ICLR_ADAPT_DIR: Path = DATA_DIR / "iclr_adapt"
OSIRIS_DIR: Path = DATA_DIR / "osiris"

# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------
MODEL_DIR: Path = PROJECT_ROOT / "models" / "saved_models"

# ---------------------------------------------------------------------------
# Reports & logs
# ---------------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_FILE: Path = LOG_DIR / "siliconpilot.log"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Data-loading behaviour
# ---------------------------------------------------------------------------
SAMPLE_MODE: bool = True          # Use small sample data during dev/testing
MAX_DOWNLOAD_GB: float = 1.0      # Hard cap on dataset downloads (gigabytes)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.2            # Fraction of data reserved for testing

# ---------------------------------------------------------------------------
# Dashboard configuration
# ---------------------------------------------------------------------------
DASHBOARD_CONFIG: dict = {
    "title": "SiliconPilot AI — Analog IC Layout Analyser",
    "theme": "dark",
    "port": 8050,
    "debug": False,
    "suppress_callback_exceptions": True,
    "update_interval_ms": 5_000,   # live-update polling interval
    "default_page": "overview",
    "sidebar_width": 240,
    "color_palette": {
        "primary": "#0057B7",
        "secondary": "#FFD700",
        "background": "#1A1A2E",
        "surface": "#16213E",
        "text": "#E0E0E0",
        "accent": "#00B4D8",
        "success": "#2ECC71",
        "warning": "#F39C12",
        "error": "#E74C3C",
    },
}

# ---------------------------------------------------------------------------
# Ensure critical directories exist at import time
# ---------------------------------------------------------------------------
for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SAMPLE_DIR,
           ICLR_ADAPT_DIR, OSIRIS_DIR, MODEL_DIR, REPORTS_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
