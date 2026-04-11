from __future__ import annotations

from vkr_classifier.config import get_settings
from vkr_classifier.training import generate_training_assets


if __name__ == "__main__":
    generate_training_assets(get_settings(), force=True)
