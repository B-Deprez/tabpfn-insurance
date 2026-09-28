"""Q1 — Severity benchmark.

Runs 5-fold CV for GLM, XGBoost, and TabPFN on French and Belgian MTPL
severity data.  Records per-fold and pooled Gamma deviance in
``res/results_severity.csv`` and RMSE / MAE / Pearson / Spearman in
``res/results_error_severity.csv``, then prints the Table 1 summary to the log.

The two correlations are stored alongside RMSE/MAE because they probe a
different failure mode: a severity model can be poorly calibrated on the
absolute scale (large RMSE / deviance) yet still rank observations correctly,
which the correlations expose.

TabPFN severity uses TabPFN's built-in ``1_plus_log`` target transform and
returns the predictive mean on the original scale (see ``TabPFNSev``).

Usage:
    python scripts/run_q1_severity.py                        # all models, append
    python scripts/run_q1_severity.py --results-tag log1p    # from scratch: all models
                                                            # into NEW _log1p files
    python scripts/run_q1_severity.py --skip-baselines \
        --results-tag log1p                                  # TabPFN re-run into NEW files
                                                            # res/results_severity_log1p.csv
                                                            # (+ _error_), seeded with the
                                                            # GLM/XGBoost rows verbatim; the
                                                            # old files are left untouched

--results-tag TAG writes to ``res/results_severity_TAG.csv`` and
``res/results_error_severity_TAG.csv`` instead of the default files. With
--skip-baselines, a tagged file that does not exist yet is seeded with the
header and the GLM/XGBoost rows of the untagged file (byte for byte); an
existing tagged file is appended to. Mirrors ``run_q2_frequency.py``.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import logging
from pathlib import Path

import numpy as np
import yaml

import os
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

try:
    import torch
except ImportError:  # CPU-only environments without torch
    torch = None


def _release_gpu() -> None:
    """Drop Python garbage and return CUDA buffers to the caching allocator.

    Called between folds and between datasets to prevent fragmentation-driven
    OOM on long sweeps. Safe no-op when torch or CUDA is absent.
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
from src.data.preprocessing import encode_features, get_raw_features, get_targets
from src.methods.glm_model import make_glm
from src.methods.xgboost_model import make_xgboost
from src.methods.tabpfn_model import make_tabpfn
from src.utils.metrics import (
    gamma_deviance,
    mae,
    pearson_corr,
    pooled_gamma_deviance,
    rmse,
    spearman_corr,
)
from src.utils.results import append_results, build_result_row

CFG_PATH = PROJECT_ROOT / "config" / "experiment_q1_severity.yaml"
RESULTS_PATH = PROJECT_ROOT / "res" / "results_severity.csv"
ERROR_METRICS_PATH = PROJECT_ROOT / "res" / "results_error_severity.csv"
logger = logging.getLogger(__name__)


def _model_family(model_name: str) -> str:
    """Return encoding family for GLM/XGBoost; TabPFN uses raw features."""
    if model_name == "glm":
        return "glm"
    if model_name == "xgboost":
        return "tree"
    return "tabpfn"  # raw unencoded features


def _expand_models(models: list[str], tabpfn_versions: list[str]) -> list[tuple[str, str]]:
    """Flatten ``models`` into (model_name, tabpfn_version) pairs.

    For ``"tabpfn"`` one pair is emitted per entry in ``tabpfn_versions``
    so each version produces an independent fit + result-row stream.
    Non-TabPFN models get a single pair with an empty version string.
    """
    pairs: list[tuple[str, str]] = []
    for m in models:
        if m == "tabpfn":
            for v in tabpfn_versions:
                pairs.append((m, v))
        else:
            pairs.append((m, ""))
    return pairs


def run_dataset(dataset: str, cfg: dict, run_baselines_flag: bool = True) -> None:
    """Run all models on one dataset for the severity task.

    ``run_baselines_flag=False`` skips GLM/XGBoost (their existing rows are
    preserved verbatim during a TabPFN-only re-run).
    """
    task = cfg["task"]  # "sev"
    experiment_id = cfg["experiment_id"]
    n_folds = cfg["cv_folds"]
    cv_seed = cfg["cv_seed"]
    tabpfn_max = cfg["tabpfn"]["max_train_size"]
    # Accept either a list (``tabpfn_versions: [...]``) or a single string
    # (``tabpfn_version: "v3"``) for backward compat with older configs.
    tabpfn_versions: list[str] = cfg["tabpfn"].get(
        "tabpfn_versions",
        [cfg["tabpfn"].get("tabpfn_version", "v3")],
    )

    logger.info("=" * 70)
    logger.info("Dataset: %s  |  Task: %s", dataset, task)
    logger.info("=" * 70)

    # ── Load & validate ───────────────────────────────────────────────────────
    df = load_dataset(dataset, task)
    validate_dataset(df, dataset, task)

    # ── CV splits ─────────────────────────────────────────────────────────────
    splits = make_cv_splits(df, dataset, cfg)

    feat_cfg = yaml.safe_load(open(PROJECT_ROOT / "config" / "features.yaml"))

    for model_name, row_version in _expand_models(cfg["models"], tabpfn_versions):
        if not run_baselines_flag and model_name in _BASELINE_MODELS:
            logger.info("--- Skipping %s (preserved verbatim) ---", model_name)
            continue
        family = _model_family(model_name)
        descr = f"{model_name} ({row_version})" if row_version else model_name
        logger.info("--- Model: %s ---", descr)

        fold_deviances: list[float] = []
        all_y: list[np.ndarray] = []
        all_mu: list[np.ndarray] = []
        all_w: list[np.ndarray] = []
        result_rows: list[dict] = []
        error_rows: list[dict] = []

        for fold in range(n_folds):
            train_df, test_df = get_fold(df, splits, fold)

            if family == "tabpfn":
                X_train = get_raw_features(train_df, dataset, feat_cfg)
                X_test = get_raw_features(test_df, dataset, feat_cfg)
            else:
                X_train, X_test = encode_features(train_df, test_df, dataset, family, feat_cfg)
            y_train, w_train, log_exp_train = get_targets(train_df, dataset, task)
            y_test, w_test, log_exp_test = get_targets(test_df, dataset, task)

            # Instantiate model
            if model_name == "glm":
                model = make_glm(task)
            elif model_name == "xgboost":
                model = make_xgboost(task)
            else:
                model = make_tabpfn(
                    task,
                    max_train_size=tabpfn_max,
                    tabpfn_version=row_version,
                )

            # Fit + predict with wall-clock timing
            fold_seed = cv_seed + fold
            t0 = time.perf_counter()
            if model_name == "tabpfn":
                # TabPFN severity is UNWEIGHTED at estimation: TabPFN supports no
                # sample_weight, so ClaimNb is NOT passed as a fit weight (the old
                # sample_weight=ClaimNb was a silent no-op). The ClaimNb weighting
                # legitimately remains at EVALUATION (gamma_deviance below).
                fit_kwargs: dict = dict(log_exposure=log_exp_train, fold_seed=fold_seed)
            else:
                # GLM (var_weights) and XGBoost (sample_weight) weight by ClaimNb.
                fit_kwargs = dict(sample_weight=w_train, log_exposure=log_exp_train)
            model.fit(X_train, y_train, **fit_kwargs)
            mu_test = model.predict(X_test, log_exposure=log_exp_test)
            elapsed = time.perf_counter() - t0

            # Evaluate
            dev = gamma_deviance(y_test, mu_test, sample_weight=w_test)
            fold_rmse = rmse(y_test, mu_test)
            fold_mae = mae(y_test, mu_test)
            fold_pearson = pearson_corr(y_test, mu_test, sample_weight=w_test)
            fold_spearman = spearman_corr(y_test, mu_test)
            fold_deviances.append(dev)
            all_y.append(y_test)
            all_mu.append(mu_test)
            all_w.append(w_test)

            logger.info(
                "  Fold %d: gamma_deviance=%.6f  rmse=%.6f  mae=%.6f  "
                "pearson=%.4f  spearman=%.4f  time=%.1fs",
                fold, dev, fold_rmse, fold_mae,
                fold_pearson, fold_spearman, elapsed,
            )

            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "gamma_deviance", dev, tabpfn_version=row_version,
            ))
            result_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "fit_predict_seconds", elapsed, tabpfn_version=row_version,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "rmse", fold_rmse, tabpfn_version=row_version,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "mae", fold_mae, tabpfn_version=row_version,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "pearson_corr", fold_pearson, tabpfn_version=row_version,
            ))
            error_rows.append(build_result_row(
                experiment_id, dataset, model_name, task, fold,
                "spearman_corr", fold_spearman, tabpfn_version=row_version,
            ))

            # Release the fitted model and fold-scoped arrays before the next
            # fold loads. Critical for TabPFN on GPU — the CUDA caching
            # allocator otherwise fragments across folds and can segfault.
            del model, train_df, test_df, X_train, X_test, mu_test
            del y_train, w_train, log_exp_train, log_exp_test
            _release_gpu()

        # Pooled OOF scores
        pooled = pooled_gamma_deviance(all_y, all_mu, all_w)
        result_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "gamma_deviance", pooled, tabpfn_version=row_version,
        ))

        y_pool = np.concatenate(all_y)
        mu_pool = np.concatenate(all_mu)
        w_pool = np.concatenate(all_w)
        pooled_pearson = pearson_corr(y_pool, mu_pool, sample_weight=w_pool)
        pooled_spearman = spearman_corr(y_pool, mu_pool)
        error_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "pearson_corr", pooled_pearson, tabpfn_version=row_version,
        ))
        error_rows.append(build_result_row(
            experiment_id, dataset, model_name, task, "pooled",
            "spearman_corr", pooled_spearman, tabpfn_version=row_version,
        ))

        mean_dev = float(np.mean(fold_deviances))
        std_dev = float(np.std(fold_deviances, ddof=1))
        logger.info(
            "  %s | %s | %s : mean=%.6f  std=%.6f  pooled=%.6f  "
            "pooled_pearson=%.4f  pooled_spearman=%.4f",
            dataset, model_name, task, mean_dev, std_dev, pooled,
            pooled_pearson, pooled_spearman,
        )

        append_results(result_rows, output_path=RESULTS_PATH)
        append_results(error_rows, output_path=ERROR_METRICS_PATH)

    # End-of-dataset cleanup before the next dataset loads.
    del df, splits, feat_cfg
    _release_gpu()


# ──────────────────────────────────────────────────────────────────────────────
# Tagged re-run: keep the old results side by side (mirrors run_q2_frequency.py)
# ──────────────────────────────────────────────────────────────────────────────

_BASELINE_MODELS = ("glm", "xgboost")


def _is_baseline_row(line: str) -> bool:
    # ``model`` is field index 3; result values never contain commas.
    return line.split(",", 5)[3] in _BASELINE_MODELS


def _tagged_path(path: Path, tag: str) -> Path:
    """``res/results_severity.csv`` -> ``res/results_severity_<tag>.csv``."""
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def _seed_baselines(src: Path, dst: Path) -> None:
    """Create ``dst`` holding the header + GLM/XGBoost rows of ``src`` verbatim.

    Text-level copy (no pandas reformatting), so the tagged file carries exactly
    the same baselines as the original. An existing ``dst`` is left as-is.
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q1 severity benchmark.")
    p.add_argument(
        "--skip-baselines", action="store_true",
        help="Do not run GLM/XGBoost this invocation (preserve existing rows).",
    )
    p.add_argument(
        "--results-tag", default=None, metavar="TAG",
        help="Write to res/results_severity_TAG.csv and "
             "res/results_error_severity_TAG.csv instead of the default files "
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
    logger.info("Starting Q1 severity benchmark")
    logger.info("Config: %s", CFG_PATH)

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
        # Baselines preserved (not recomputed): copy them into the tagged file.
        if args.skip_baselines:
            for src, dst in zip(untagged, (RESULTS_PATH, ERROR_METRICS_PATH)):
                _seed_baselines(src, dst)

    for dataset in cfg["datasets"]:
        run_dataset(dataset, cfg, run_baselines_flag=not args.skip_baselines)
        _release_gpu()

    logger.info(
        "Q1 complete — deviances appended to %s, error metrics to %s",
        RESULTS_PATH, ERROR_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
