import numpy as np
import pandas as pd
import pytest

from tradingagents.macro import betas
from tradingagents.macro.config import FACTORS

pytestmark = pytest.mark.unit


def _factor_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    data = {f: rng.normal(0, 0.01, n) for f in FACTORS}
    return pd.DataFrame(data, index=idx)


def test_recovers_known_betas_on_synthetic_data():
    fac = _factor_frame()
    true_b = {f: float(v) for f, v in zip(FACTORS, [1.5, -0.8, -2.0, 0.3, 1.0, 0.5])}
    noise = np.random.default_rng(1).normal(0, 1e-6, len(fac))
    stock_ret = sum(true_b[f] * fac[f] for f in FACTORS) + noise
    out = betas.compute_betas("TEST", stock_ret, fac)
    for f in FACTORS:
        assert abs(out.betas[f] - true_b[f]) < 0.05
    assert out.r2 > 0.99
    assert out.confidence == "high"


def test_short_history_is_shrunk_and_flagged_low():
    fac = _factor_frame(n=40)
    stock_ret = 2.0 * fac["mkt"] + np.random.default_rng(2).normal(0, 0.001, len(fac))
    out = betas.compute_betas("SHORT", stock_ret, fac)
    assert out.confidence == "low"
    assert abs(out.betas["mkt"]) < 2.0          # shrunk toward zero
    assert out.n_obs == 40


def test_build_factor_returns_shapes_and_columns():
    idx = pd.bdate_range("2025-01-01", periods=10)
    raw = {
        "tnx": pd.Series(np.linspace(4.0, 4.2, 10), index=idx),     # level → diff
        "dxy": pd.Series(np.linspace(100, 102, 10), index=idx),     # level → ret
        "hy": pd.Series(np.linspace(3.0, 3.1, 10), index=idx),
        "oil": pd.Series(np.linspace(70, 72, 10), index=idx),
        "spy": pd.Series(np.linspace(500, 510, 10), index=idx),
        "iwf": pd.Series(np.linspace(300, 305, 10), index=idx),
        "iwd": pd.Series(np.linspace(180, 181, 10), index=idx),
    }
    fac = betas.build_factor_returns(raw)
    assert list(fac.columns) == FACTORS
    assert len(fac) == 9                          # one row lost to differencing
    assert all(abs(fac[col].std(ddof=0) - 1.0) < 1e-4 for col in FACTORS)  # standardized (float tol)


def test_linear_shrink_zone_ramps_between_floor_and_full():
    # n=156 is mid-ramp: t=(156-60)/192=0.5 → k=0.25+0.75*0.5=0.625
    fac = _factor_frame(n=156)
    stock_ret = 2.0 * fac["mkt"] + np.random.default_rng(3).normal(0, 1e-6, len(fac))
    out = betas.compute_betas("MID", stock_ret, fac)
    assert out.confidence == "low"
    assert out.n_obs == 156
    implied_k = out.betas["mkt"] / 2.0
    assert 0.55 < implied_k < 0.70   # ~0.625
