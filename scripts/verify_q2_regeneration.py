"""Post-regeneration invariant check for the Q2 frequency exposure fix.

Run AFTER ``run_q2_frequency.py --force ...`` (e.g. on the VSC). Confirms the
GLM/XGBoost rows in the live result CSVs are byte-identical to the copy that
``--force`` just archived in ``res/archive/`` — i.e. the baselines were preserved
verbatim and only the TabPFN rows were regenerated. Also summarises the corrected
TabPFN rows now present.

Usage:
    python scripts/verify_q2_regeneration.py

Exit 0 = baselines unchanged; 1 = drift (investigate before trusting the run).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "res"
ARCHIVE = RES / "archive"
BASELINE_MODELS = ("glm", "xgboost")
PAIRS = ["results_frequency.csv", "results_error_frequency.csv"]


def _baseline_lines(path: Path) -> list[str]:
    body = path.read_text().splitlines(keepends=True)[1:]  # drop header
    return sorted(ln for ln in body if ln.split(",", 5)[3] in BASELINE_MODELS)


def _latest_archive(stem: str, suffix: str) -> Path | None:
    cands = sorted(
        ARCHIVE.glob(f"{stem}_*{suffix}"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def main() -> int:
    ok = True
    for name in PAIRS:
        live = RES / name
        stem, suffix = live.stem, live.suffix
        arch = _latest_archive(stem, suffix)
        if not live.exists():
            print(f"!! {name}: live file missing")
            ok = False
            continue
        if arch is None:
            print(f"!! {name}: no timestamped archive found in {ARCHIVE} "
                  f"(did you run with --force?)")
            ok = False
            continue

        live_base = _baseline_lines(live)
        arch_base = _baseline_lines(arch)
        n_tabpfn = sum(
            1 for ln in live.read_text().splitlines()[1:]
            if "tabpfn" in ln.split(",", 5)[3]
        )
        if live_base == arch_base:
            print(f"OK  {name}: {len(live_base)} GLM/XGBoost rows byte-identical "
                  f"to archive {arch.name}; {n_tabpfn} TabPFN rows present.")
        else:
            ok = False
            print(f"!! {name}: baseline rows DIFFER from archive {arch.name} "
                  f"(live={len(live_base)} vs archive={len(arch_base)})")

    print("\n" + ("OK — baselines preserved verbatim." if ok else "DRIFT — do not trust the run."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
