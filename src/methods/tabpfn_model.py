"""TabPFN wrappers for insurance pricing (frequency, severity, and binary classification).

TabPFN handles mixed data types (strings, integers, floats) and missing
values natively — no feature encoding is required.  Raw feature columns from
``get_raw_features()`` are passed directly to the model.

The internal ``ModelVersion`` (``v2_5``, ``v2_6`` or ``v3``) is selected per
call via ``create_default_for_version``.  The default is ``v3``.

Exposure / target handling:
    Frequency  : exposure-as-feature (EAJ Referee 1 fix). TabPFN supports
                 neither a Poisson offset nor sample_weight, so exposure is an
                 ordinary input feature; the model is trained on the bounded
                 COUNT (ClaimNb) and the annualised rate μ is recovered with a
                 counterfactual Exposure=1.0 query at predict time. This is NOT a
                 true offset — the model learns the exposure effect rather than
                 having a unit slope on log(Exposure) imposed.
    Severity   : response = raw AvgSeverity with TabPFN's built-in ``1_plus_log``
                 target transform (fit on log(1 + y)); ``predict`` returns the
                 mean of the predictive distribution on the ORIGINAL scale, so
                 no manual exp() back-transform (which would give roughly the
                 median, biased below the mean).

TabPFNRegressor.fit() accepts only (X, y) — no sample_weight argument.

Device auto-detection order: CUDA → MPS → CPU.

Training is capped at ``max_train_size`` rows.  When the training fold
exceeds this limit a random subsample is drawn (seed = fold_seed).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapping of public string labels → tabpfn.constants.ModelVersion enum members.
# Resolved lazily inside helpers so importing this module does not require tabpfn.
_VERSION_LABELS = ("v2_5", "v2_6", "v3")

# Upstream pretraining-context ceilings enforced by TabPFN. Inputs larger than
# this raise ``TabPFNValidationError`` unless ``ignore_pretraining_limits=True``
# is passed.  Versions not listed here have no fixed ceiling.
_VERSION_MAX_TRAIN_SIZE = {"v2_5": 50_000, "v2_6": 100_000}

# Default name of the injected exposure feature for the frequency task. Frequency
# runners that build features with ``get_raw_features_freq`` already carry the
# exposure column and pass its real name (``Exposure`` / ``expo``) via
# ``exposure_col``. Callers that instead supply exposure through ``sample_weight``
# (e.g. the SHAP path) get it injected under this default name.
_DEFAULT_EXPOSURE_FEATURE = "Exposure"


def _effective_max_train_size(version: str, requested: int) -> int:
    """Return ``min(requested, upstream_ceiling)`` for the given version."""
    cap = _VERSION_MAX_TRAIN_SIZE.get(version)
    return min(requested, cap) if cap is not None else requested


def _resolve_model_version(version: str):
    """Translate ``"v2_5"`` / ``"v2_6"`` / ``"v3"`` to the corresponding ``ModelVersion`` enum."""
    from tabpfn.constants import ModelVersion
    if version == "v2_5":
        return ModelVersion.V2_5
    if version == "v2_6":
        return ModelVersion.V2_6
    if version == "v3":
        return ModelVersion.V3
    raise ValueError(
        f"Unknown tabpfn_version {version!r}; expected one of {_VERSION_LABELS}"
    )


def _make_regressor(version: str, device: str, **overrides):
    """Build a ``TabPFNRegressor`` for the requested internal model version.

    ``overrides`` are forwarded to ``create_default_for_version`` (e.g. the
    severity ``inference_config``); without them the version defaults apply.
    """
    from tabpfn import TabPFNRegressor
    return TabPFNRegressor.create_default_for_version(
        _resolve_model_version(version), device=device, **overrides,
    )


# Severity target transform: TabPFN fits on log(1 + y) and inverts the whole
# predictive distribution, so ``predict`` returns E[y] on the original scale.
# Same setting as the Prior Labs insurance cookbook's claim-amount models.
_SEV_INFERENCE_CONFIG = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("1_plus_log",)}


def _make_classifier(version: str, device: str):
    """Build a ``TabPFNClassifier`` for the requested internal model version."""
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier.create_default_for_version(
        _resolve_model_version(version), device=device,
    )


def _detect_device() -> str:
    """Return the best available device string for TabPFN."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _subsample(
    X: pd.DataFrame,
    y: np.ndarray,
    max_size: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (X_sub, y_sub) subsampled to at most ``max_size`` rows."""
    n = len(X)
    if n <= max_size:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_size, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]


class TabPFNFreq:
    """TabPFN regressor for frequency modelling (exposure as an input feature).

    Exposure fairness fix (EAJ Referee 1). TabPFN supports neither a Poisson
    offset (as the GLM uses) nor sample_weight / base_margin (as XGBoost uses),
    so exposure cannot enter the way it does for the baselines. Instead:

    * **Exposure is an ordinary input feature.** The model is trained on the
      *bounded* claim COUNT ``ClaimNb`` — never the rate ``ClaimNb/Exposure``,
      which explodes for tiny exposures — with exposure as one of the inputs.
    * **The annualised rate is recovered by a counterfactual query.**
      ``predict`` sets the exposure feature to ``1.0`` for every row, so the
      model's expected-count output is already the expected count over a full
      year = the annualised frequency μ — exactly the rate scale that the
      exposure-weighted Poisson deviance expects, so it plugs into the existing
      metric unchanged.

    Limitation: exposure-as-feature is NOT a true offset. GLM/XGBoost impose a
    unit slope on ``log(Exposure)``; TabPFN *learns* the exposure effect from the
    data. It is the fairest option available for a model with neither an offset
    nor sample_weight, but it is not identical to an offset.

    Exposure may reach the model two ways, both handled here:
      1. already present in ``X_train`` as ``exposure_col`` (the frequency runner
         builds features with ``get_raw_features_freq``); or
      2. supplied via ``sample_weight`` while absent from ``X_train`` (the SHAP
         path passes raw features plus exposure as the weight) — it is then
         injected as the ``exposure_col`` feature.
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
        exposure_col: str = _DEFAULT_EXPOSURE_FEATURE,
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self.exposure_col = exposure_col
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNFreq: device=%s, max_train_size=%d, version=%s, exposure_col=%s",
            self._device, max_train_size, tabpfn_version, exposure_col,
        )

    def _with_exposure(
        self,
        X: pd.DataFrame,
        exposure: "np.ndarray | float",
    ) -> pd.DataFrame:
        """Return a copy of ``X`` with the exposure feature set to ``exposure``.

        ``exposure`` is a per-row array at training time, or the scalar ``1.0``
        for the counterfactual annualisation query at predict time. Assigning an
        existing column overwrites it in place (position preserved); assigning a
        new column appends it last. Training and prediction therefore see the
        exposure feature in the same position either way.
        """
        X_aug = X.copy()
        X_aug[self.exposure_col] = exposure
        return X_aug

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNFreq":
        """Fit TabPFN on raw features + exposure, targeting the claim COUNT.

        Args:
            X_train: raw (unencoded) feature DataFrame. If it already contains
                ``exposure_col`` it is used as-is; otherwise the exposure is taken
                from ``sample_weight`` and injected as that feature.
            y_train: claim COUNTS (ClaimNb) — modelled directly, NOT converted to
                a rate and NOT log-transformed (counts are small and bounded).
            sample_weight: exposure array; used to populate the exposure feature
                when it is not already a column of ``X_train``.
            log_exposure: unused; accepted for interface parity with GLM/XGBoost.
            fold_seed: RNG seed for the subsample draw.
        """
        if self.exposure_col in X_train.columns:
            X_aug = X_train
        else:
            exposure = (
                sample_weight if sample_weight is not None else np.ones(len(y_train))
            )
            X_aug = self._with_exposure(X_train, np.asarray(exposure, dtype=float))

        self._feature_names = list(X_aug.columns)

        # Model the bounded count directly — no rate, no log transform.
        y_count = np.asarray(y_train, dtype=float)

        X_sub, y_sub = _subsample(X_aug, y_count, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_aug):
            logger.info(
                "TabPFNFreq: subsampled %d → %d rows (seed=%d)",
                len(X_aug), len(X_sub), fold_seed,
            )

        self._model = _make_regressor(self.tabpfn_version, self._device)
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNFreq fitted on %d rows (target=ClaimNb count)", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the annualised claim rate μ via a counterfactual Exposure=1 query.

        Every test row is evaluated at a full year of exposure, so the model's
        expected-count output is already the per-year rate. ``log_exposure`` is
        accepted for interface parity but unused — the counterfactual query, not
        an offset, does the annualisation.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        X_query = self._with_exposure(X_test, 1.0)
        return self._model.predict(X_query)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNSev:
    """TabPFN regressor for severity modelling (``1_plus_log`` target transform).

    Raw unencoded features are passed directly to TabPFN. The response is the
    raw AvgSeverity; TabPFN's built-in ``1_plus_log`` transform fits on
    log(1 + y) and maps the predictive distribution back, so ``predict`` returns
    its MEAN on the original scale. This replaces the former manual
    fit-on-log(y) / exp(prediction), which returned exp(E[log y]) — roughly the
    median, systematically below E[y] for right-skewed claim amounts.

    Unweighted at estimation: TabPFN supports no sample_weight, so the ClaimNb
    weighting used by the GLM/XGBoost severity models is NOT applied at fit time
    (it legitimately remains at evaluation, where gamma_deviance weights by
    ClaimNb). ``sample_weight`` is still accepted for interface parity but ignored.
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNSev: device=%s, max_train_size=%d, version=%s",
            self._device, max_train_size, tabpfn_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNSev":
        """Fit TabPFN on raw average severity with the ``1_plus_log`` transform.

        Args:
            X_train: raw (unencoded) feature DataFrame.
            y_train: average severity (AvgSeverity = ClaimAmount / ClaimNb),
                passed on the original scale — TabPFN applies log(1 + y).
            sample_weight: accepted for interface parity; not used (TabPFN
                does not support sample_weight in fit()).
            log_exposure: unused for severity; accepted for interface parity.
            fold_seed: RNG seed for the subsample draw.
        """
        self._feature_names = list(X_train.columns)
        y = np.asarray(y_train, dtype=float)

        X_sub, y_sub = _subsample(X_train, y, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_train):
            logger.info(
                "TabPFNSev: subsampled %d → %d rows (seed=%d)",
                len(X_train), len(X_sub), fold_seed,
            )

        self._model = _make_regressor(
            self.tabpfn_version, self._device, inference_config=_SEV_INFERENCE_CONFIG,
        )
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNSev fitted on %d rows", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted average severity: the predictive MEAN, original scale."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict(X_test)

    def get_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return SHAP values (original severity scale) from TabPFN's explainer."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        if hasattr(self._model, "get_shap_values"):
            return self._model.get_shap_values(X_test)
        logger.warning("TabPFN built-in SHAP not available; falling back to KernelExplainer")
        import shap
        bg = X_test.iloc[:min(100, len(X_test))]
        explainer = shap.KernelExplainer(self._model.predict, bg)
        return explainer.shap_values(X_test, nsamples=100)

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNClf:
    """TabPFN classifier for binary classification tasks.

    Raw unencoded features are passed directly to TabPFN.
    ``predict()`` returns the probability of class 1 (claim).
    """

    def __init__(
        self,
        max_train_size: int = 100_000,
        tabpfn_version: str = "v3",
    ) -> None:
        self.max_train_size = _effective_max_train_size(tabpfn_version, max_train_size)
        self.tabpfn_version = tabpfn_version
        self._model = None
        self._feature_names: list[str] = []
        self._device = _detect_device()
        logger.info(
            "TabPFNClf: device=%s, max_train_size=%d, version=%s",
            self._device, max_train_size, tabpfn_version,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        log_exposure: np.ndarray | None = None,
        fold_seed: int = 0,
    ) -> "TabPFNClf":
        """Fit TabPFN classifier on raw features.

        Args:
            X_train: raw (unencoded) feature DataFrame.
            y_train: binary labels (0/1).
            sample_weight: accepted for interface parity; not used.
            log_exposure: accepted for interface parity; not used.
            fold_seed: RNG seed for the subsample draw.
        """
        self._feature_names = list(X_train.columns)
        y_int = y_train.astype(int)

        X_sub, y_sub = _subsample(X_train, y_int, self.max_train_size, fold_seed)
        if len(X_sub) < len(X_train):
            logger.info(
                "TabPFNClf: subsampled %d → %d rows (seed=%d)",
                len(X_train), len(X_sub), fold_seed,
            )

        self._model = _make_classifier(self.tabpfn_version, self._device)
        self._model.fit(X_sub, y_sub)
        logger.info("TabPFNClf fitted on %d rows", len(X_sub))
        return self

    def predict(
        self,
        X_test: pd.DataFrame,
        log_exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return predicted claim probabilities (probability of class 1)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        return self._model.predict_proba(X_test)[:, 1]

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names


class TabPFNFreqWithShap(TabPFNFreq):
    """TabPFNFreq extended with SHAP computation (used by run_q3_shap.py).

    The explained frame is the counterfactual Exposure=1.0 query, so it matches
    the feature set the model was trained on (raw features + exposure). Note this
    means the returned SHAP array now includes an exposure column — a downstream
    change for Q3 frequency interpretability, flagged for a separate decision.
    """

    def get_shap_values(self, X_test: pd.DataFrame) -> np.ndarray:
        """Return SHAP values from TabPFN's built-in explainer or KernelExplainer."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet")
        X_query = self._with_exposure(X_test, 1.0)
        if hasattr(self._model, "get_shap_values"):
            return self._model.get_shap_values(X_query)
        logger.warning("TabPFN built-in SHAP not available; falling back to KernelExplainer")
        import shap
        bg = X_query.iloc[:min(100, len(X_query))]
        explainer = shap.KernelExplainer(self._model.predict, bg)
        return explainer.shap_values(X_query, nsamples=100)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_tabpfn(
    task: str,
    max_train_size: int = 100_000,
    shap: bool = False,
    tabpfn_version: str = "v3",
    exposure_col: str = _DEFAULT_EXPOSURE_FEATURE,
):
    """Return the appropriate TabPFN wrapper for the given task.

    Args:
        task: ``"freq"``, ``"sev"``, or ``"clf"``.
        max_train_size: maximum training rows.
        shap: if True, return a SHAP-capable variant for Q3 (freq/sev only).
        tabpfn_version: internal ``ModelVersion`` to use — ``"v2_5"``, ``"v2_6"`` or ``"v3"``.
        exposure_col: (freq only) name of the exposure input feature. Frequency
            runners pass the dataset's real exposure column (``Exposure`` /
            ``expo``); callers that supply exposure via ``sample_weight`` can rely
            on the default.
    """
    if task == "freq":
        cls = TabPFNFreqWithShap if shap else TabPFNFreq
        return cls(max_train_size, tabpfn_version=tabpfn_version, exposure_col=exposure_col)
    if task == "sev":
        return TabPFNSev(max_train_size, tabpfn_version=tabpfn_version)
    if task == "clf":
        return TabPFNClf(max_train_size, tabpfn_version=tabpfn_version)
    raise ValueError(f"Unknown task '{task}'")
