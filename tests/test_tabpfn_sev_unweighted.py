"""Unit tests for TabPFN severity (Step 3 cleanup + Step 6 1_plus_log fix).

Proves — deterministically, with no real TabPFN — that:
  * dropping the dead ``sample_weight=ClaimNb`` from the TabPFN severity fit
    cannot change the numbers: the model is trained on exactly the same inputs
    with or without the weight (it was always ignored);
  * the target handling uses TabPFN's built-in ``1_plus_log`` transform: the raw
    AvgSeverity is passed to fit, the regressor is built with that
    inference_config, and predict returns TabPFN's (original-scale) mean as-is.

The sample_weight removal alone is number-preserving; the 1_plus_log change is
not (it replaces exp(E[log y]) with E[y]), so severity is re-run on the VSC.

Runnable directly (``python tests/test_tabpfn_sev_unweighted.py``); also
collectable by pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.methods import tabpfn_model
from src.methods.tabpfn_model import TabPFNSev


class _Recorder:
    """Fake regressor: records the exact (X, y) it is fit on; predict is a stub."""

    def __init__(self, const: float = 2.0, overrides: dict | None = None) -> None:
        self.const = const
        self.overrides = overrides or {}
        self.fit_X: pd.DataFrame | None = None
        self.fit_y: np.ndarray | None = None

    def fit(self, X, y):
        self.fit_X = X.copy()
        self.fit_y = np.asarray(y, dtype=float).copy()
        return self

    def predict(self, X):
        return np.full(len(X), self.const, dtype=float)


class _patch_make_regressor:
    def __init__(self, const: float = 2.0) -> None:
        self.const = const
        self._orig = None
        self.last = None

    def __enter__(self):
        self._orig = tabpfn_model._make_regressor

        def _fake(version, device, **overrides):
            self.last = _Recorder(self.const, overrides)
            return self.last

        tabpfn_model._make_regressor = _fake
        return self

    def __exit__(self, *exc):
        tabpfn_model._make_regressor = self._orig
        return False


def _toy(n: int = 30, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "VehPower": rng.integers(4, 10, size=n),
        "DrivAge": rng.integers(20, 70, size=n),
    })
    avg_sev = rng.uniform(500.0, 5000.0, size=n)      # AvgSeverity target
    claimnb = rng.integers(1, 4, size=n).astype(float)  # ClaimNb weight
    return X, avg_sev, claimnb


def test_severity_sample_weight_is_inert():
    """Fitting with vs without sample_weight trains on identical inputs."""
    X, y, w = _toy()
    with _patch_make_regressor() as p1:
        TabPFNSev().fit(X, y, sample_weight=w, fold_seed=0)
        Xw, yw = p1.last.fit_X.copy(), p1.last.fit_y.copy()
    with _patch_make_regressor() as p2:
        TabPFNSev().fit(X, y, fold_seed=0)   # no sample_weight
        Xn, yn = p2.last.fit_X.copy(), p2.last.fit_y.copy()
    # Same training frame and same target either way.
    pd.testing.assert_frame_equal(Xw, Xn)
    np.testing.assert_array_equal(yw, yn)


def test_severity_uses_builtin_1_plus_log():
    """Raw AvgSeverity is fit with the 1_plus_log config; predict is not exp'd."""
    X, y, w = _toy()
    with _patch_make_regressor(const=2000.0) as p:
        model = TabPFNSev().fit(X, y, fold_seed=0)
        pred = model.predict(X)
    np.testing.assert_allclose(p.last.fit_y, y)          # raw target, no manual log
    assert p.last.overrides == {
        "inference_config": {"REGRESSION_Y_PREPROCESS_TRANSFORMS": ("1_plus_log",)},
    }
    np.testing.assert_allclose(pred, 2000.0)              # TabPFN mean returned as-is


def _all_tests():
    return [test_severity_sample_weight_is_inert, test_severity_uses_builtin_1_plus_log]


def main() -> int:
    passed = failed = 0
    for t in _all_tests():
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as a:
            print(f"FAIL  {t.__name__}: {a}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
