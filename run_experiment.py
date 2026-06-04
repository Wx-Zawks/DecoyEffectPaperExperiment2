from __future__ import annotations

import json
import sys
from pathlib import Path

from repro.experiment import run_paper_reproduction


def load_overrides(argv: list[str]) -> dict:
    if len(argv) < 2:
        return {}
    config_path = Path(argv[1]).resolve()
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    overrides = load_overrides(sys.argv)
    result = run_paper_reproduction(overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
