"""Post-regeneration invariant check for the TabPFN re-runs (Q2 freq, Q1 sev).

Run AFTER the re-run. Confirms every GLM/XGBoost statistical value in the new
result CSVs matches the reference copy (timestamp and wall-clock
``fit_predict_seconds`` excluded), i.e. the baselines are unchanged, whether they
were copied (``--skip-baselines``) or recomputed (from-scratch run). Also counts
the TabPFN rows now present.

Copied baselines are byte-identical. Recomputed ones are only equal up to float
noise (GLM ~1e-11 relative; XGBoost ``hist`` up to ~1e-6 across machines/threads),
so values are compared with a relative tolerance (``--rtol``, default 1e-5) and
the largest relative difference is reported.

For a from-scratch VSC run the reference files only exist on the Mac: copy the
tagged CSVs back into the Mac's ``res/`` and run this there.

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


def _baseline_values(path: Path) -> dict[tuple, str]:
    """GLM/XGBoost rows keyed by identity, wall-clock rows skipped.

    Fields: timestamp, experiment_id, dataset, model, task, fold, metric, value,
    tabpfn_version. Key = all fields except timestamp and value; value kept as text.
    """
    out: dict[tuple, str] = {}
    for ln in path.read_text().splitlines()[1:]:  # drop header
        f = ln.split(",")
        if f[3] in BASELINE_MODELS and f[6] != "fit_predict_seconds":
            key = (*f[1:7], *f[8:])
            if key in out:
                raise SystemExit(f"!! {path.name}: duplicate baseline row {key}")
            out[key] = f[7]
    return out


def _rel_diff(a: str, b: str) -> float:
    x, y = float(a), float(b)
    return 0.0 if x == y else abs(x - y) / max(abs(x), abs(y))


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
    p.add_argument(
        "--rtol", type=float, default=1e-5,
        help="Max relative difference per value (default 1e-5).",
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

        live_base = _baseline_values(live)
        ref_base = _baseline_values(ref)
        n_tabpfn = sum(
            1 for ln in live.read_text().splitlines()[1:]
            if "tabpfn" in ln.split(",", 5)[3]
        )
        if live_base.keys() != ref_base.keys():
            ok = False
            missing = sorted(ref_base.keys() - live_base.keys())
            extra = sorted(live_base.keys() - ref_base.keys())
            print(f"!! {live.name}: baseline rows differ from {ref.name} "
                  f"({len(missing)} missing, {len(extra)} extra), "
                  f"e.g. {(missing or extra)[:2]}")
            continue

        diffs = {k: _rel_diff(live_base[k], ref_base[k]) for k in ref_base}
        worst = max(diffs, key=diffs.get)
        n_exact = sum(d == 0.0 for d in diffs.values())
        summary = (f"{len(diffs)} GLM/XGBoost values vs {ref.name}: {n_exact} identical, "
                   f"max rel diff {diffs[worst]:.1e} ({worst[2]} {worst[5]} fold {worst[4]}); "
                   f"{n_tabpfn} TabPFN rows present.")
        if diffs[worst] <= args.rtol:
            print(f"OK  {live.name}: {summary}")
        else:
            ok = False
            print(f"!! {live.name}: {summary} Exceeds --rtol {args.rtol:g}.")

    print("\n" + ("OK — baselines unchanged." if ok else "DRIFT — do not trust the run."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
