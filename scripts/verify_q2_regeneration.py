"""Post-regeneration invariant check for the TabPFN re-runs (Q2 freq, Q1 sev).

Run AFTER the exposure re-run (e.g. on the VSC). Confirms the GLM/XGBoost rows
in the new result CSVs are byte-identical to the reference copy, i.e. the
baselines were preserved verbatim and only the TabPFN rows were regenerated.
Also counts the TabPFN rows now present.

Reference depends on how the re-run was written:
  * ``--results-tag TAG`` : new ``res/results_frequency_TAG.csv`` (+ ``_error_``)
                            vs the untouched original ``res/results_frequency.csv``.
  * ``--force``           : live ``res/results_frequency.csv`` vs the timestamped
                            copy ``--force`` just archived in ``res/archive/``.

Usage:
    python scripts/verify_q2_regeneration.py --results-tag expo               # Q2 frequency
    python scripts/verify_q2_regeneration.py --task sev --results-tag log1p   # Q1 severity
    python scripts/verify_q2_regeneration.py                                  # --force layout

Exit 0 = baselines unchanged; 1 = drift (investigate before trusting the run).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "res"
ARCHIVE = RES / "archive"
BASELINE_MODELS = ("glm", "xgboost")
PAIRS = {
    "freq": ["results_frequency.csv", "results_error_frequency.csv"],
    "sev": ["results_severity.csv", "results_error_severity.csv"],
}


def _baseline_lines(path: Path) -> list[str]:
    body = path.read_text().splitlines(keepends=True)[1:]  # drop header
    return sorted(ln for ln in body if ln.split(",", 5)[3] in BASELINE_MODELS)


def _latest_archive(stem: str, suffix: str) -> Path | None:
    cands = sorted(
        ARCHIVE.glob(f"{stem}_*{suffix}"),
        key=lambda p: p.stat().st_mtime,
    )
    return cands[-1] if cands else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify baselines were preserved.")
    p.add_argument(
        "--task", choices=sorted(PAIRS), default="freq",
        help="Which result files to check (default: freq).",
    )
    p.add_argument(
        "--results-tag", default=None, metavar="TAG",
        help="Check the tagged files written with --results-tag TAG against the "
             "untagged originals (default: --force archive layout).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    tag = args.results_tag
    ok = True
    for name in PAIRS[args.task]:
        stem, suffix = Path(name).stem, Path(name).suffix
        if tag:
            live = RES / f"{stem}_{tag}{suffix}"
            ref = RES / name
        else:
            live = RES / name
            ref = _latest_archive(stem, suffix)
        if not live.exists():
            print(f"!! {live.name}: live file missing")
            ok = False
            continue
        if ref is None or not ref.exists():
            hint = f"{RES / name} missing" if tag else (
                f"no timestamped archive in {ARCHIVE} (did you run with --force?)"
            )
            print(f"!! {live.name}: no reference file ({hint})")
            ok = False
            continue

        live_base = _baseline_lines(live)
        ref_base = _baseline_lines(ref)
        n_tabpfn = sum(
            1 for ln in live.read_text().splitlines()[1:]
            if "tabpfn" in ln.split(",", 5)[3]
        )
        if live_base == ref_base:
            print(f"OK  {live.name}: {len(live_base)} GLM/XGBoost rows byte-identical "
                  f"to {ref.name}; {n_tabpfn} TabPFN rows present.")
        else:
            ok = False
            print(f"!! {live.name}: baseline rows DIFFER from {ref.name} "
                  f"(live={len(live_base)} vs reference={len(ref_base)})")

    print("\n" + ("OK — baselines preserved verbatim." if ok else "DRIFT — do not trust the run."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
