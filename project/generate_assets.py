from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vkr_classifier.config import get_settings  # noqa: E402
from vkr_classifier.training import generate_training_assets  # noqa: E402


if __name__ == "__main__":
    generate_training_assets(get_settings(), force=True)

