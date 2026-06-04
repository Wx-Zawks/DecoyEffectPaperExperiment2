from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repro.experiment import _build_figures
from repro.svg import ensure_dir, write_dashboard


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs_py/paper_results.json")
    resolved_results_path = results_path.resolve()
    results = json.loads(resolved_results_path.read_text(encoding="utf-8"))
    figure_dir = resolved_results_path.parent / "figures"
    ensure_dir(figure_dir)
    figure_paths = _build_figures(figure_dir, results)
    write_dashboard(
        "DIM + PRM 论文复现实验面板",
        [(item["label"], item["filename"]) for item in figure_paths],
        figure_dir / "index.html",
    )
    print(figure_dir)


if __name__ == "__main__":
    main()
