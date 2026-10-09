"""Q2 — Frequency ceiling.

Evaluates TabPFN at subsample sizes {2k, 5k, 10k} and compares against
GLM and XGBoost baselines on both French and Belgian MTPL frequency data.

Exposure handling (EAJ Referee 1 fix). GLM uses a log-exposure offset and
XGBoost a log-exposure base_margin; TabPFN supports neither, so for TabPFN
exposure is an **input feature** (via ``get_raw_features_freq``): the model is
trained on the bounded COUNT ``ClaimNb`` and the annualised rate μ is recovered
by a counterfactual query at Exposure=1.0. μ plugs into the same
exposure-weighted Poisson-deviance call as the baselines — the metric is
unchanged. (Not a true offset: TabPFN learns the exposure effect rather than
having a unit slope on log(Exposure) imposed.)

Subsampling is nested: the script draws max(subsample_sizes) training rows
per fold (seed = cv_seed + fold) and reuses the first N rows for each
smaller size (2k ⊂ 5k ⊂ 10k), keeping the comparison fair.

Per-fold and pooled Poisson deviance are written to
``res/results_frequency.csv``; the exposure-weighted RMSE-on-rate metric is
written to ``res/results_error_frequency.csv``.

Usage:
    python scripts/run_q2_frequency.py                       # baselines + TabPFN, append
    python scripts/run_q2_frequency.py --results-tag expo \
        --tabpfn-versions v2_6                               # from scratch: GLM/XGBoost +
                                                            # TabPFN v2_6 into NEW _expo files
    python scripts/run_q2_frequency.py --skip-baselines \
        --results-tag expo --tabpfn-versions v2_6            # exposure re-run into NEW files
                                                            # res/results_frequency_expo.csv
                                                            # (+ _error_), seeded with the
                                                            # GLM/XGBoost rows verbatim; the
                                                            # old files are left untouched
    python scripts/run_q2_frequency.py --skip-baselines \
        --results-tag expo --tabpfn-versions v3              # append TabPFN v3 to the same
    python scripts/run_q2_frequency.py --force \
        --tabpfn-versions v2_6                               # archive + rewrite IN PLACE

--results-tag TAG writes to ``res/results_frequency_TAG.csv`` and
``res/results_error_frequency_TAG.csv`` instead of the default files, so a re-run
sits next to the old results for comparison. With --skip-baselines, a tagged file
that does not exist yet is seeded with the header and the GLM/XGBoost rows of the
untagged file (byte for byte), making it a drop-in replacement for the original;
an existing tagged file is appended to.

--force archives the (possibly tagged) frequency CSVs to ``res/archive/`` and
rewrites them keeping only the GLM/XGBoost rows (byte for byte); it implies
--skip-baselines so the baselines are preserved, never recomputed.
--skip-baselines alone appends TabPFN rows without archiving or touching the
baselines.
"""

from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

try:
    import torch
except ImportError:  # CPU-only environments without torch
    torch = None


def _release_gpu() -> None:
    """Drop Python garbage and return CUDA buffers to the caching allocator.

    Called between TabPFN fits to prevent fragmentation-driven OOM on long
    multi-fold / multi-size sweeps. Safe no-op when torch or CUDA is absent.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_setup import setup_logging
from src.data.loaders import load_dataset
from src.data.contracts import validate_dataset
from src.data.cv import make_cv_splits, get_fold
from src.data.preprocessing import (
    encode_features,
    get_freq_exposure_col,
    get_raw_features,
    get_raw_features_freq,
    get_targets,
)
from src.methods.glm_model import make_glm
from src.methods.xgboost_model import make_xgboost
from src.methods.tabpfn_model import (
    TabPFNFreq,
    _VERSION_MAX_TRAIN_SIZE,
    _make_regressor,
)
from src.utils.metrics import (
    exposure_weighted_rmse_rate,
    poisson_deviance,
    pooled_poisson_deviance,
)
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q2_frequency.yaml"
RESULTS_PATH = PROJECT_ROOT / "res" / "results_frequency.csv"
ERROR_METRICS_PATH = PROJECT_ROOT / "res" / "results_error_frequency.csv"
logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    return "glm" if model_name == "glm" else "tree"


def run_baselines(
    dataset: str,
    cfg: dict,
    df,
    splits: list,
    feat_cfg: dict,
) -> None:
    """Run GLM and XGBoost on the full training fold (no subsampling)."""
    task = cfg["task"]
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]

    for model_name in ("glm", "xgboost"):
        family = _model_family(model_name)
        logger.info("--- Baseline: %s ---", model_name)

        fold_deviances: list[float] = []
        all_y: list[np.ndarray] = []
        all_mu: list[np.ndarray] = []
        all_e: list[np.ndarray] = []
        result_rows: list[dict] = []
        error_rows: list[dict] = []

        for fold in range(n_folds):
            train_df, test_df = get_fold(df, splits, fold)

            X_train, X_test = encode_features(train_df, test_df, dataset, family, feat_cfg)
            y_train, w_train, log_exp_train = get_targets(train_df, dataset, task)
            y_test, w_test, log_exp_test = get_targets(test_df, dataset, task)

            model = make_glm(task) if model_name == "glm" else make_xgboost(task)

            t0 = time.perf_counter()
            model.fit(X_train, y_train, sample_weight=w_train, log_exposure=log_exp_train)
            mu_test = model.predict(X_test, log_exposure=log_exp_test)
            elapsed = time.perf_counter() - t0

            dev = poisson_deviance(y_test, mu_test, w_test, sample_weight=w_test)
            y_rate_test = y_test / np.maximum(w_test, 1e-10)
            fold_rmse_rate = exposure_weighted_rmse_rate(y_rate_test, mu_test, w_test)
            fold_deviances.append(dev)
            all_y.append(y_test)
            all_mu.append(mu_test)
            all_e.append(w_test)

            logger.info(
                "  Fold %d: poisson_deviance=%.6f  exposure_weighted_rmse_rate=%.6f  time=%.1fs",
                fold, dev, fold_rmse_rate, elapsed,
            )

            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "poisson_deviance", dev,
            ))
            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "fit_predict_seconds", elapsed,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "exposure_weighted_rmse_rate", fold_rmse_rate,
            ))

            del model, train_df, test_df, X_train, X_test, mu_test
            del y_train, w_train, log_exp_train, log_exp_test, y_rate_test

        pooled = pooled_poisson_deviance(all_y, all_mu, all_e, all_e)
        result_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "poisson_deviance", pooled,
        ))

        logger.info(
            "  %s | %s | %s : mean=%.6f  std=%.6f  pooled=%.6f",
            dataset, model_name, task,
            float(np.mean(fold_deviances)),
            float(np.std(fold_deviances, ddof=1)),
            pooled,
        )
        append_results(result_rows, output_path=RESULTS_PATH)
        append_results(error_rows, output_path=ERROR_METRICS_PATH)


def run_tabpfn_subsample(
    dataset: str,
    cfg: dict,
    df,
    splits: list,
    feat_cfg: dict,
    versions_override: list[str] | None = None,
) -> None:
    """Run TabPFN at each subsample size, with nested subsampling.

    ``versions_override`` (from ``--tabpfn-versions``) restricts the run to a
    subset of the configured TabPFN versions, e.g. v2_6 in one step and v3 in a
    later one.
    """
    task = cfg["task"]
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]
    # Parse subsample_sizes: integers feed the nested-subsample sweep, while
    # the "full" sentinel triggers an additional pass on the entire training
    # fold (no subsampling).  Putting "full" last keeps memory growth monotonic.
    subsample_sizes_raw = cfg["tabpfn"]["subsample_sizes"]
    numeric_sizes: list[int] = sorted(
        s for s in subsample_sizes_raw if isinstance(s, int)
    )
    include_full: bool = "full" in subsample_sizes_raw
    sizes_to_run: list[int | str] = [*numeric_sizes]
    if include_full:
        sizes_to_run.append("full")

    # Accept either a list or a single string for backward compat.
    tabpfn_versions: list[str] = cfg["tabpfn"].get(
        "tabpfn_versions",
        [cfg["tabpfn"].get("tabpfn_version", "v3")],
    )
    if versions_override is not None:
        unknown = [v for v in versions_override if v not in tabpfn_versions]
        if unknown:
            logger.warning(
                "  --tabpfn-versions %s not in configured versions %s; running anyway",
                unknown, tabpfn_versions,
            )
        tabpfn_versions = list(versions_override)
    max_size = numeric_sizes[-1] if numeric_sizes else 0

    logger.info(
        "--- TabPFN subsample sweep: sizes=%s, versions=%s ---",
        sizes_to_run, tabpfn_versions,
    )

    # Hoist device detection out of the per-size loop. The regressor is built
    # via the shared factory so the v2_6/v3 dispatch matches the other scripts.
    from src.methods.tabpfn_model import _detect_device
    device = _detect_device()

    for fold in range(n_folds):
        train_df, test_df = get_fold(df, splits, fold)

        # Exposure-as-feature fix (EAJ Referee 1): raw features PLUS the exposure
        # column, so TabPFN sees exposure as an ordinary input.
        exp_col = get_freq_exposure_col(dataset, feat_cfg)
        X_train_full = get_raw_features_freq(train_df, dataset, feat_cfg)
        X_test = get_raw_features_freq(test_df, dataset, feat_cfg)
        y_train_full, w_train_full, _ = get_targets(train_df, dataset, task)
        y_test, w_test, _ = get_targets(test_df, dataset, task)
        del train_df, test_df

        # Model the bounded COUNT (ClaimNb) directly — NOT the rate. y_train_full
        # is already ClaimNb; keep it as the fit target and slice per size.
        y_count_full = np.asarray(y_train_full, dtype=float)

        # Counterfactual test frame: every policy at a FULL year of exposure, so
        # the model's expected-count output IS the annualised rate μ.
        X_test_cf = X_test.copy()
        X_test_cf[exp_col] = 1.0

        # Draw the master subsample (nested: smaller sizes take first N rows).
        # When "full" is requested we permute the entire fold so that the
        # smaller subsamples remain strict subsets of the full pass.
        fold_seed = cv_seed + fold
        rng = np.random.default_rng(fold_seed)
        n_available = len(X_train_full)
        master_size = n_available if include_full else min(max_size, n_available)
        master_idx = rng.choice(n_available, size=master_size, replace=False)

        y_rate_test = y_test / np.maximum(w_test, 1e-10)

        for size in sizes_to_run:
            if size == "full":
                actual_size = n_available
                model_label = "tabpfn_full"
                X_sub = X_train_full
                y_sub = y_count_full
            else:
                actual_size = min(size, master_size)
                model_label = f"tabpfn_{actual_size}"
                idx = master_idx[:actual_size]
                X_sub = X_train_full.iloc[idx].reset_index(drop=True)
                y_sub = y_count_full[idx]

            # v2.5 and v2.6 have fixed pretraining ceilings (50k and 100k rows
            # respectively); the "full" data point exists to exercise v3's
            # expanded context size, so skip the older versions whenever the
            # subsample exceeds their supported limit.
            versions_for_size = [
                v for v in tabpfn_versions
                if not (
                    v in _VERSION_MAX_TRAIN_SIZE
                    and actual_size > _VERSION_MAX_TRAIN_SIZE[v]
                )
            ]
            skipped = [v for v in tabpfn_versions if v not in versions_for_size]
            if skipped:
                logger.info(
                    "  Fold %d | size=%s (n=%d): skipping %s (>100k pretraining limit)",
                    fold, model_label, actual_size, skipped,
                )

            # Both versions fit the same subsample, so per-version timing
            # and deviance can be compared directly.
            for current_version in versions_for_size:
                t0 = time.perf_counter()
                model = _make_regressor(current_version, device)
                model.fit(X_sub, y_sub)                 # fit on ClaimNb counts
                mu_test = model.predict(X_test_cf)       # Exposure=1.0 → annualised μ
                elapsed = time.perf_counter() - t0

                dev = poisson_deviance(y_test, mu_test, w_test, sample_weight=w_test)
                fold_rmse_rate = exposure_weighted_rmse_rate(y_rate_test, mu_test, w_test)
                size_label = "full" if size == "full" else str(actual_size)
                logger.info(
                    "  Fold %d | size=%s (n=%d) | %s: poisson_deviance=%.6f  "
                    "exposure_weighted_rmse_rate=%.6f  time=%.1fs",
                    fold, size_label, actual_size, current_version,
                    dev, fold_rmse_rate, elapsed,
                )

                # Model label encodes the size (or "full"); the per-version
                # axis lives in the ``tabpfn_version`` column.
                rows = [
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "poisson_deviance", dev, tabpfn_version=current_version,
                    ),
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "fit_predict_seconds", elapsed,
                        tabpfn_version=current_version,
                    ),
                ]
                error_rows = [
                    build_result_row(
                        experiment_id, dataset, model_label, task, fold,
                        "exposure_weighted_rmse_rate", fold_rmse_rate,
                        tabpfn_version=current_version,
                    ),
                ]
                append_results(rows, output_path=RESULTS_PATH)
                append_results(error_rows, output_path=ERROR_METRICS_PATH)

                # Critical: release the fitted TabPFN and its GPU buffers
                # before the next (size, version) iteration. Without this,
                # the CUDA caching allocator fragments across the
                # sizes × versions × n_folds fits and eventually segfaults.
                del model, mu_test
                _release_gpu()

            del X_sub, y_sub

        # End-of-fold cleanup before loading the next fold's data.
        del X_train_full, X_test, X_test_cf, y_train_full, w_train_full, y_test, w_test
        del y_count_full, y_rate_test, master_idx
        _release_gpu()


def run_dataset(
    dataset: str,
    cfg: dict,
    run_baselines_flag: bool = True,
    versions_override: list[str] | None = None,
) -> None:
    """Run baseline and TabPFN experiments on one dataset.

    ``run_baselines_flag=False`` skips GLM/XGBoost (their existing rows are
    preserved verbatim during a TabPFN-only regeneration).
    """
    task = cfg["task"]
    logger.info("=" * 70)
    logger.info("Dataset: %s  |  Task: %s", dataset, task)
    logger.info("=" * 70)

    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)
    splits = make_cv_splits(df, dataset, cfg)
    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    if run_baselines_flag:
        run_baselines(dataset, cfg, df, splits, feat_cfg)
    else:
        logger.info("--- Skipping GLM/XGBoost baselines (preserved verbatim) ---")
    run_tabpfn_subsample(dataset, cfg, df, splits, feat_cfg, versions_override)

    del df, splits, feat_cfg
    _release_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Regeneration: archive + keep baselines (decision A — see CLAUDE.md)
# ──────────────────────────────────────────────────────────────────────────────

_BASELINE_MODELS = ("glm", "xgboost")


def _is_baseline_row(line: str) -> bool:
    # ``model`` is field index 3; result values never contain commas.
    return line.split(",", 5)[3] in _BASELINE_MODELS


def _tagged_path(path: Path, tag: str) -> Path:
    """``res/results_frequency.csv`` -> ``res/results_frequency_<tag>.csv``."""
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def _seed_baselines(src: Path, dst: Path) -> None:
    """Create ``dst`` holding the header + GLM/XGBoost rows of ``src`` verbatim.

    Used by --results-tag so the tagged file carries exactly the same baselines
    as the original (text-level copy, no pandas reformatting) and can replace it
    one-for-one. An existing ``dst`` is left as-is, so a later stage (v3 after
    v2_6) appends onto it.
    """
    if dst.exists():
        logger.info("--results-tag: %s exists; appending to it", dst.name)
        return
    if not src.exists():
        # Fail fast: continuing would write a tagged file with no GLM/XGBoost rows.
        raise SystemExit(
            f"--results-tag: {src} does not exist, so there are no baselines to copy "
            f"into {dst.name}. Drop --skip-baselines to compute them in this run."
        )
    lines = src.read_text().splitlines(keepends=True)
    header, kept = lines[:1], [ln for ln in lines[1:] if _is_baseline_row(ln)]
    dst.write_text("".join(header + kept))
    logger.info(
        "--results-tag: seeded %s with %d baseline rows from %s",
        dst.name, len(kept), src.name,
    )


def _archive_and_keep_baselines(path: Path) -> None:
    """Archive ``path`` to res/archive/ then rewrite it keeping only GLM/XGBoost.

    The kept rows are written back byte-for-byte (text-level filter, no pandas
    reformatting) so the preserved baselines stay identical to the snapshot. The
    stale TabPFN rows are dropped; the corrected TabPFN rows are appended
    afterwards by the normal (append-only) writer.
    """
    if not path.exists():
        logger.warning("--force: %s does not exist; nothing to archive", path)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = PROJECT_ROOT / "res" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, archive_path)

    lines = path.read_text().splitlines(keepends=True)
    header, body = lines[:1], lines[1:]
    kept = [ln for ln in body if _is_baseline_row(ln)]
    path.write_text("".join(header + kept))
    logger.info(
        "--force: archived %s -> %s ; kept %d baseline rows, dropped %d TabPFN rows",
        path.name, archive_path, len(kept), len(body) - len(kept),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q2 frequency ceiling experiment.")
    p.add_argument(
        "--force", action="store_true",
        help="Archive the frequency result CSVs and drop stale TabPFN rows, "
             "keeping GLM/XGBoost rows verbatim; implies --skip-baselines.",
    )
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Do not run GLM/XGBoost this invocation (preserve existing rows).",
    )
    p.add_argument(
        "--tabpfn-versions", nargs="+", default=None, metavar="VERSION",
        help="Restrict the TabPFN sweep to these versions (e.g. v2_6). "
             "Defaults to the versions in the config.",
    )
    p.add_argument(
        "--results-tag", default=None, metavar="TAG",
        help="Write to res/results_frequency_TAG.csv and "
             "res/results_error_frequency_TAG.csv instead of the default files "
             "(the old results stay untouched). With --skip-baselines a new tagged "
             "file is seeded with the GLM/XGBoost rows of the default file.",
    )
    return p.parse_args()


def main() -> None:
    global RESULTS_PATH, ERROR_METRICS_PATH
    args = _parse_args()

    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg["experiment_id"])
    logger.info("Starting Q2 frequency ceiling experiment")
    if args.tabpfn_versions:
        logger.info("TabPFN versions restricted to: %s", args.tabpfn_versions)

    # --force implies preserving (not recomputing) the baselines.
    skip_baselines = args.skip_baselines or args.force

    # Redirect output to tagged files so the old results stay side by side.
    if args.results_tag:
        untagged = (RESULTS_PATH, ERROR_METRICS_PATH)
        RESULTS_PATH, ERROR_METRICS_PATH = (
            _tagged_path(p, args.results_tag) for p in untagged
        )
        logger.info(
            "--results-tag %s: writing to %s and %s",
            args.results_tag, RESULTS_PATH.name, ERROR_METRICS_PATH.name,
        )

    if args.force:
        logger.info("--force: regenerating TabPFN frequency rows (archive + rewrite)")
        for path in (RESULTS_PATH, ERROR_METRICS_PATH):
            _archive_and_keep_baselines(path)

    # Baselines preserved (not recomputed): copy them into a new tagged file.
    if args.results_tag and skip_baselines:
        for src, dst in zip(untagged, (RESULTS_PATH, ERROR_METRICS_PATH)):
            _seed_baselines(src, dst)

    for dataset in cfg["datasets"]:
        run_dataset(
            dataset, cfg,
            run_baselines_flag=not skip_baselines,
            versions_override=args.tabpfn_versions,
        )
        _release_gpu()

    logger.info(
        "Q2 complete — deviances appended to %s, error metrics to %s",
        RESULTS_PATH, ERROR_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
