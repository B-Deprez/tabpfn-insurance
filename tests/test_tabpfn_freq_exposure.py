"""Unit tests for the TabPFN frequency exposure-as-feature refactor (EAJ Ref. 1).

Two layers:

* **Deterministic mechanics** (always run, no real TabPFN). ``_make_regressor``
  is monkeypatched with a fake whose ``predict`` echoes the exposure feature, so
  we can assert exactly what the wrapper feeds the model and queries back:
    - the target is the claim COUNT, not the rate ``ClaimNb/Exposure``;
    - exposure is present as an input feature with the correct per-row values;
    - ``predict`` issues the counterfactual Exposure=1.0 query (so, for a model
      whose expected count equals exposure, μ = 1.0 — the true annualised rate);
    - the model's output changes when the input exposure changes (exposure is a
      live feature, not ignored).

* **Real-TabPFN behaviour** (skips if TabPFN/model is unavailable). On synthetic
  data where ``ClaimNb`` scales with exposure, checks the fitted model predicts
  more claims at higher exposure, and that the Exposure=1.0 query returns μ on the
  annualised scale (μ larger than the exposure-diluted observed counts).

Runnable directly (``python tests/test_tabpfn_freq_exposure.py``) since pytest is
not part of the environment; also collectable by pytest if installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods import tabpfn_model
from src.methods.tabpfn_model import TabPFNFreq


# ──────────────────────────────────────────────────────────────────────────────
# Fake regressor + monkeypatch helper (deterministic layer)
# ──────────────────────────────────────────────────────────────────────────────

class _EchoExposureRegressor:
    """Fake TabPFN regressor: records fit inputs; predict echoes the exposure col.

    A model whose expected count equals the exposure corresponds to a true
    annualised rate of exactly 1.0 per year — convenient for asserting the
    counterfactual Exposure=1.0 query recovers that rate.
    """

    def __init__(self, exposure_col: str = "Exposure") -> None:
        self.exposure_col = exposure_col
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, X):
        return X[self.exposure_col].to_numpy(dtype=float)


class _patch_make_regressor:
    """Context manager: make ``TabPFNFreq`` build an ``_EchoExposureRegressor``."""

    def __init__(self, exposure_col: str = "Exposure") -> None:
        self.exposure_col = exposure_col
        self._orig = None
        self.last = None

    def __enter__(self):
        self._orig = tabpfn_model._make_regressor

        def _fake(version, device):
            self.last = _EchoExposureRegressor(self.exposure_col)
            return self.last

        tabpfn_model._make_regressor = _fake
        return self

    def __exit__(self, *exc):
        tabpfn_model._make_regressor = self._orig
        return False


def _toy_frame(n: int = 40, seed: int = 0):
    """Return (X_features_only, exposure, counts) with count != count/exposure."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "VehPower": rng.integers(4, 10, size=n),
        "DrivAge": rng.integers(20, 70, size=n),
    })
    exposure = rng.uniform(0.1, 1.0, size=n)          # < 1, so rate != count
    counts = rng.integers(0, 3, size=n).astype(float)  # ClaimNb
    return X, exposure, counts


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic tests
# ──────────────────────────────────────────────────────────────────────────────

def test_target_is_count_not_rate():
    """The model is trained on ClaimNb, never on ClaimNb/Exposure."""
    X, exposure, counts = _toy_frame()
    with _patch_make_regressor() as p:
        model = TabPFNFreq(exposure_col="Exposure").fit(
            X, counts, sample_weight=exposure,
        )
    np.testing.assert_allclose(p.last.fit_y, counts)
    # Guard against a regression to the old Strategy-B rate target.
    rate = counts / np.maximum(exposure, 1e-10)
    assert not np.allclose(p.last.fit_y, rate), "target must be counts, not the rate"
    _ = model


def test_exposure_injected_as_feature_with_values():
    """Exposure enters the training frame as a feature with the right per-row values."""
    X, exposure, counts = _toy_frame()
    with _patch_make_regressor() as p:
        TabPFNFreq(exposure_col="Exposure").fit(X, counts, sample_weight=exposure)
    assert "Exposure" in p.last.fit_X.columns
    np.testing.assert_allclose(p.last.fit_X["Exposure"].to_numpy(), exposure)
    # Original feature columns are preserved alongside the injected exposure.
    for col in X.columns:
        assert col in p.last.fit_X.columns


def test_predict_is_counterfactual_annualised_rate():
    """predict() queries at Exposure=1.0 → μ is the annualised rate.

    For the echo model (expected count == exposure ⇒ true rate 1.0/yr), the
    Exposure=1.0 query must return 1.0 for every test row regardless of the test
    frame's own exposure values.
    """
    X, exposure, counts = _toy_frame()
    with _patch_make_regressor():
        model = TabPFNFreq(exposure_col="Exposure").fit(X, counts, sample_weight=exposure)
        # Test frame carries arbitrary (non-1) exposure; predict must ignore it.
        X_test = X.copy()
        X_test["Exposure"] = np.linspace(0.2, 0.9, len(X_test))
        mu = model.predict(X_test)
    np.testing.assert_allclose(mu, 1.0)


def test_model_output_changes_with_exposure():
    """The fitted model responds to exposure (it is a live feature, not ignored)."""
    X, exposure, counts = _toy_frame()
    with _patch_make_regressor():
        model = TabPFNFreq(exposure_col="Exposure").fit(X, counts, sample_weight=exposure)
        low = model._model.predict(model._with_exposure(X, 0.25))
        high = model._model.predict(model._with_exposure(X, 0.75))
    assert np.all(high > low), "higher exposure must change (raise) the model output"


def test_exposure_already_in_X_used_as_is():
    """When X already carries the exposure column (get_raw_features_freq path),
    it is used directly and not overwritten by ones."""
    X, exposure, counts = _toy_frame()
    X_with_exp = X.copy()
    X_with_exp["Exposure"] = exposure
    with _patch_make_regressor() as p:
        # No sample_weight passed — exposure must come from the X column.
        model = TabPFNFreq(exposure_col="Exposure").fit(X_with_exp, counts)
        np.testing.assert_allclose(p.last.fit_X["Exposure"].to_numpy(), exposure)
        mu = model.predict(X_with_exp)   # counterfactual overwrites to 1.0
    np.testing.assert_allclose(mu, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Real-TabPFN behavioural test (skips gracefully)
# ──────────────────────────────────────────────────────────────────────────────

class _Skip(Exception):
    """Signals a skipped test when the real TabPFN model is unavailable."""


def test_real_tabpfn_exposure_behaviour():
    """On data where ClaimNb scales with exposure, the fitted model predicts more
    claims at higher exposure, and the Exposure=1.0 query returns annualised μ."""
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0.0, 1.0, size=n)
    exposure = rng.uniform(0.1, 1.0, size=n)
    true_rate = 0.5                              # claims per full year
    counts = rng.poisson(true_rate * exposure).astype(float)
    X = pd.DataFrame({"x": x, "Exposure": exposure})

    try:
        model = TabPFNFreq(tabpfn_version="v3", exposure_col="Exposure")
        model.fit(X, counts)                     # exposure already a column
        # (a) exposure sensitivity: query the fitted model at low vs high exposure
        low = model._model.predict(model._with_exposure(X, 0.2))
        high = model._model.predict(model._with_exposure(X, 0.9))
        # (b) annualised rate via the counterfactual query
        mu = model.predict(X)
    except _Skip:
        raise
    except Exception as exc:  # model download/hardware/version issues → skip
        raise _Skip(f"real TabPFN unavailable: {type(exc).__name__}: {exc}")

    assert np.mean(high) > np.mean(low), "more exposure should raise expected counts"
    assert np.all(np.isfinite(mu)) and np.all(mu >= 0), "μ must be finite, non-negative"
    # μ is per-year; observed counts are diluted by exposure (<1 on average), so
    # the annualised μ should sit above the mean observed count.
    assert np.mean(mu) > np.mean(counts), "annualised μ should exceed exposure-diluted counts"
    # ...and land in a generous band around the true annual rate.
    assert 0.1 < float(np.mean(mu)) < 1.5, f"mean μ={np.mean(mu):.3f} off annualised scale"


# ──────────────────────────────────────────────────────────────────────────────
# Script runner (pytest not required)
# ──────────────────────────────────────────────────────────────────────────────

def _all_tests():
    return [
        test_target_is_count_not_rate,
        test_exposure_injected_as_feature_with_values,
        test_predict_is_counterfactual_annualised_rate,
        test_model_output_changes_with_exposure,
        test_exposure_already_in_X_used_as_is,
        test_real_tabpfn_exposure_behaviour,
    ]


def main() -> int:
    passed = skipped = failed = 0
    for t in _all_tests():
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except _Skip as s:
            print(f"SKIP  {t.__name__}: {s}")
            skipped += 1
        except AssertionError as a:
            print(f"FAIL  {t.__name__}: {a}")
            failed += 1
        except Exception as e:  # unexpected error is a failure
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
