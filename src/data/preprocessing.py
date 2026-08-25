"""Feature encoding — fold-by-fold preprocessing.

All encoding parameters (category orders, label maps) are fit on the training
fold and applied to both train and test folds.  The authoritative per-feature
encoding decisions live in ``config/features.yaml``.

Public API
----------
encode_features(train_df, test_df, dataset, model_family, feat_cfg=None)
    → (X_train, X_test)  — encoded feature DataFrames

get_raw_features(df, dataset, feat_cfg=None)
    → pd.DataFrame  — unencoded feature columns for TabPFN-2.6

get_feature_names(dataset, model_family, feat_cfg=None)
    → list[str]  — column names that encode_features will produce (train only)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CFG_PATH = PROJECT_ROOT / "config" / "features.yaml"


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_feat_cfg() -> dict:
    with open(FEATURES_CFG_PATH) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Per-column encoders
# ──────────────────────────────────────────────────────────────────────────────

def _encode_numeric(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pass through a numeric column unchanged."""
    return (
        train_series.rename(name).to_frame(),
        test_series.rename(name).to_frame(),
    )


def _encode_binary(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
    mapping: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map string (or object) values to 0/1 integers.

    The mapping is defined in features.yaml.  Values are cast to str before
    lookup to handle the ``fleet`` column whose values may already be integers.
    """
    str_mapping = {str(k): int(v) for k, v in mapping.items()}

    def apply(s: pd.Series) -> pd.DataFrame:
        encoded = s.astype(str).map(str_mapping)
        if encoded.isna().any():
            unseen = set(s.astype(str).unique()) - set(str_mapping)
            raise ValueError(
                f"Binary column '{name}': unmapped values {unseen}. "
                f"Known keys: {list(str_mapping)}"
            )
        return encoded.astype(int).rename(name).to_frame()

    return apply(train_series), apply(test_series)


def _encode_ordinal_string_glm(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
    order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode an ordinal string for GLM (drop_first=True).

    The category order is fixed by ``order`` so that the dropped (reference)
    category is always the first element.  Test-fold dummies are aligned to the
    training columns to handle any absent categories.
    """
    cat_type = pd.CategoricalDtype(categories=order, ordered=True)

    def dummies(s: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(
            s.astype(cat_type), prefix=name, drop_first=True, dtype=float
        )

    train_d = dummies(train_series)
    test_d = dummies(test_series).reindex(columns=train_d.columns, fill_value=0.0)
    return train_d, test_d


def _encode_ordinal_string_tree(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
    order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map an ordinal string to integers respecting the declared order."""
    mapping = {cat: i for i, cat in enumerate(order)}

    def apply(s: pd.Series) -> pd.DataFrame:
        encoded = s.map(mapping)
        if encoded.isna().any():
            unseen = set(s.unique()) - set(mapping)
            raise ValueError(
                f"Ordinal column '{name}': unseen categories {unseen}. "
                f"Known order: {order}"
            )
        return encoded.astype(int).rename(name).to_frame()

    return apply(train_series), apply(test_series)


def _encode_nominal_string_glm(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode a nominal string for GLM.

    Categories are derived from the training fold only (drop_first=True).
    Any test-fold category not seen in training is mapped to the all-zero
    vector (absorbed into the reference category).
    """
    cats = sorted(train_series.dropna().unique())
    cat_type = pd.CategoricalDtype(categories=cats, ordered=False)

    def dummies(s: pd.Series) -> pd.DataFrame:
        return pd.get_dummies(
            s.astype(cat_type), prefix=name, drop_first=True, dtype=float
        )

    train_d = dummies(train_series)
    test_d = dummies(test_series).reindex(columns=train_d.columns, fill_value=0.0)
    return train_d, test_d


def _encode_nominal_string_tree(
    name: str,
    train_series: pd.Series,
    test_series: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-encode a nominal string for tree models.

    The integer mapping is fit on the training fold.  Unseen test-fold
    categories are assigned code -1.
    """
    cats = sorted(train_series.dropna().unique())
    mapping = {cat: i for i, cat in enumerate(cats)}

    def apply(s: pd.Series) -> pd.DataFrame:
        encoded = s.map(mapping).fillna(-1).astype(int)
        return encoded.rename(name).to_frame()

    return apply(train_series), apply(test_series)


# ──────────────────────────────────────────────────────────────────────────────
# Main encoder
# ──────────────────────────────────────────────────────────────────────────────

def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset: str,
    model_family: str,
    feat_cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode feature columns fold-by-fold.

    All encoder parameters (categories, label maps) are derived from
    ``train_df`` only and then applied to ``test_df``.

    Args:
        train_df: training fold DataFrame (contains both feature and target cols).
        test_df: test fold DataFrame.
        dataset: ``"freMTPL2"`` or ``"beMTPL97"``.
        model_family: ``"glm"`` or ``"tree"`` (XGBoost and TabPFN both use tree).
        feat_cfg: optional pre-loaded features.yaml dict.

    Returns:
        Tuple ``(X_train, X_test)`` of encoded feature DataFrames.
    """
    if feat_cfg is None:
        feat_cfg = _load_feat_cfg()
    if model_family not in ("glm", "tree"):
        raise ValueError(f"model_family must be 'glm' or 'tree', got '{model_family}'")

    features = feat_cfg["datasets"][dataset]["features"]
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for feat in features:
        name = feat["name"]
        ftype = feat["type"]
        encoding = feat["glm"] if model_family == "glm" else feat["tree"]

        if name not in train_df.columns:
            logger.warning("Feature '%s' not found in DataFrame — skipping", name)
            continue

        tr_s = train_df[name]
        te_s = test_df[name]

        if ftype == "numeric" or encoding == "numeric":
            tr_enc, te_enc = _encode_numeric(name, tr_s, te_s)

        elif ftype == "binary" or encoding == "binary":
            tr_enc, te_enc = _encode_binary(name, tr_s, te_s, feat["mapping"])

        elif ftype == "ordinal_string":
            if encoding == "one_hot":
                tr_enc, te_enc = _encode_ordinal_string_glm(
                    name, tr_s, te_s, feat["order"]
                )
            elif encoding == "ordinal":
                tr_enc, te_enc = _encode_ordinal_string_tree(
                    name, tr_s, te_s, feat["order"]
                )
            else:
                raise ValueError(f"Unknown encoding '{encoding}' for ordinal_string '{name}'")

        elif ftype == "nominal_string":
            if encoding == "one_hot":
                tr_enc, te_enc = _encode_nominal_string_glm(name, tr_s, te_s)
            elif encoding == "label":
                tr_enc, te_enc = _encode_nominal_string_tree(name, tr_s, te_s)
            else:
                raise ValueError(f"Unknown encoding '{encoding}' for nominal_string '{name}'")

        else:
            raise ValueError(f"Unknown feature type '{ftype}' for column '{name}'")

        train_parts.append(tr_enc)
        test_parts.append(te_enc)

    X_train = pd.concat(train_parts, axis=1).reset_index(drop=True)
    X_test = pd.concat(test_parts, axis=1).reset_index(drop=True)
    logger.info(
        "encode_features [%s/%s]: %d train rows, %d test rows, %d columns",
        dataset, model_family, len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, X_test


# ──────────────────────────────────────────────────────────────────────────────
# Target and weight extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_targets(
    df: pd.DataFrame,
    dataset: str,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (y, exposure_or_weight, log_exposure) for a given dataset/task.

    Returns
    -------
    y : np.ndarray
        For freq: claim counts (ClaimNb / nclaims).
        For sev: average severity (AvgSeverity).
    weight : np.ndarray
        For freq: exposure (Exposure / expo) — used as sample_weight for
            TabPFN/XGBoost Strategy B and as offset denominator for GLM.
        For sev: claim count (ClaimNb / nclaims) — sample weight.
    log_exposure : np.ndarray
        log(exposure) for the GLM Poisson offset.  For severity tasks this is
        an array of zeros (not used).
    """
    if dataset == "freMTPL2":
        if task == "freq":
            y = df["ClaimNb"].to_numpy(dtype=float)
            weight = df["Exposure"].to_numpy(dtype=float)
        else:
            y = df["AvgSeverity"].to_numpy(dtype=float)
            weight = df["ClaimNb"].to_numpy(dtype=float)
    elif dataset == "beMTPL97":
        if task == "freq":
            y = df["nclaims"].to_numpy(dtype=float)
            weight = df["expo"].to_numpy(dtype=float)
        else:
            y = df["AvgSeverity"].to_numpy(dtype=float)
            weight = df["nclaims"].to_numpy(dtype=float)
    elif dataset == "fretelematic":
        # Binary classification: y = 0/1, no exposure weighting.
        y = df["Claim_binary"].to_numpy(dtype=float)
        weight = np.ones(len(df))
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    log_exposure = np.log(weight) if task == "freq" else np.zeros(len(df))
    return y, weight, log_exposure


# ──────────────────────────────────────────────────────────────────────────────
# Raw feature extraction for TabPFN-2.6
# ──────────────────────────────────────────────────────────────────────────────

def get_raw_features(
    df: pd.DataFrame,
    dataset: str,
    feat_cfg: dict | None = None,
) -> pd.DataFrame:
    """Return the raw (unencoded) feature columns for TabPFN-2.6.

    TabPFN-2.6 handles mixed data types (strings, integers, floats) and missing
    values natively, so no encoding is applied.  Only the feature columns
    listed in ``config/features.yaml`` are selected — target, exposure, ID, and
    derived columns are excluded automatically.

    Args:
        df: fold DataFrame (train or test).
        dataset: ``"freMTPL2"`` or ``"beMTPL97"``.
        feat_cfg: optional pre-loaded features.yaml dict.

    Returns:
        DataFrame containing only the raw feature columns, index reset.
    """
    if feat_cfg is None:
        feat_cfg = _load_feat_cfg()
    features = feat_cfg["datasets"][dataset]["features"]
    cols = [f["name"] for f in features if f["name"] in df.columns]
    missing = [f["name"] for f in features if f["name"] not in df.columns]
    if missing:
        logger.warning("get_raw_features: columns not found in DataFrame — %s", missing)
    return df[cols].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Exposure-as-feature helpers for the TabPFN FREQUENCY task
# ──────────────────────────────────────────────────────────────────────────────
# Exposure fairness fix (EAJ Referee 1): TabPFN supports neither a Poisson offset
# nor sample_weight, so exposure is handed to it as an ordinary input feature and
# the annualised rate is recovered later with a counterfactual Exposure=1.0 query.
# These helpers are used ONLY by the TabPFN frequency path; ``get_raw_features``,
# ``encode_features`` and ``get_targets`` are unchanged, so the severity, SHAP and
# GLM/XGBoost paths are unaffected.

def get_freq_exposure_col(dataset: str, feat_cfg: dict | None = None) -> str | None:
    """Return the exposure column TabPFN-frequency uses as an input feature.

    Reads the ``tabpfn_freq_exposure_feature`` key from ``config/features.yaml``.
    Returns ``None`` when the dataset declares no exposure feature (e.g. the
    classification dataset).  This key is never consulted by ``encode_features``,
    so it cannot leak into the GLM/XGBoost design matrices.
    """
    if feat_cfg is None:
        feat_cfg = _load_feat_cfg()
    return feat_cfg["datasets"][dataset].get("tabpfn_freq_exposure_feature")


def get_raw_features_freq(
    df: pd.DataFrame,
    dataset: str,
    feat_cfg: dict | None = None,
) -> pd.DataFrame:
    """Raw TabPFN features for the FREQUENCY task, with exposure appended.

    Extends :func:`get_raw_features` by adding the exposure column named in
    ``tabpfn_freq_exposure_feature`` as the last input feature.  The model is then
    trained on the bounded count (``ClaimNb``) with exposure among the inputs; the
    annualised rate μ is recovered by querying at ``Exposure=1.0`` (see
    ``TabPFNFreq``).

    Note this is NOT a true offset — TabPFN *learns* the exposure effect rather
    than having a unit slope on ``log(Exposure)`` imposed, which is the fairest
    option available given TabPFN has neither an offset nor sample_weight.
    """
    if feat_cfg is None:
        feat_cfg = _load_feat_cfg()
    exp_col = get_freq_exposure_col(dataset, feat_cfg)
    if exp_col is None:
        raise ValueError(
            f"Dataset '{dataset}' declares no 'tabpfn_freq_exposure_feature' in "
            f"features.yaml; cannot build the TabPFN frequency feature set."
        )
    if exp_col not in df.columns:
        raise ValueError(
            f"Exposure feature '{exp_col}' not found in DataFrame for '{dataset}'."
        )
    X = get_raw_features(df, dataset, feat_cfg).copy()
    # Use .to_numpy() so the append is index-agnostic (both frames are 0..n-1).
    X[exp_col] = df[exp_col].to_numpy(dtype=float)
    return X
