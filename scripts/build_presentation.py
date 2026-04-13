from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[1] / "project"
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    target = project_dir / "scripts" / "build_presentation.py"
    runpy.run_path(str(target), run_name="__main__")
