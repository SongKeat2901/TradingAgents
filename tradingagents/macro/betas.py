"""Per-stock macro betas via rolling OLS. Pure — operates on provided frames.

`build_factor_returns` constructs the standardized factor-return matrix from
raw factor series (yields → daily change; prices → daily return; growth−value
spread). `compute_betas` regresses a stock's daily returns on the factors via
numpy.linalg.lstsq and applies shrinkage for short samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FACTORS, BETA_MIN_OBS, BETA_SHRINK_FLOOR


@dataclass
class Betas:
    ticker: str
    betas: dict[str, float]
    r2: float
    confidence: str                    # "high" | "low"
    n_obs: int


def build_factor_returns(raw: dict[str, pd.Series]) -> pd.DataFrame:
    """raw keys: tnx, dxy, hy, oil, spy, iwf, iwd. Returns a DataFrame whose
    columns are exactly FACTORS, aligned on common dates, NaNs dropped."""
    cols = {
        "d_10y": raw["tnx"].diff(),
        "d_dxy": raw["dxy"].pct_change(),
        "d_hy_spread": raw["hy"].diff(),
        "oil_ret": raw["oil"].pct_change(),
        "mkt": raw["spy"].pct_change(),
        "growth_value": raw["iwf"].pct_change() - raw["iwd"].pct_change(),
    }
    f = pd.DataFrame(cols)[FACTORS].dropna()
    # Standardize each factor to unit variance (zero std → divide by 1) so betas
    # are comparable across factors and the dot-product with [-1,1] regime moves
    # is dimensionally balanced.
    std = f.std(ddof=0).replace(0, 1.0)
    return (f - f.mean()) / std


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_betas(ticker: str, stock_ret: pd.Series, factors: pd.DataFrame) -> Betas:
    df = pd.concat([stock_ret.rename("y"), factors], axis=1).dropna()
    n = len(df)
    if n < 5:
        return Betas(ticker, {f: 0.0 for f in FACTORS}, 0.0, "low", n)
    y = df["y"].to_numpy()
    X = df[FACTORS].to_numpy()
    Xc = np.column_stack([np.ones(n), X])         # intercept
    coef, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    beta_vec = coef[1:]
    yhat = Xc @ coef
    r2 = _r2(y, yhat)

    # Shrinkage: full weight at/above BETA_MIN_OBS, linearly toward 0 down to
    # BETA_SHRINK_FLOOR, heavily shrunk below.
    if n >= BETA_MIN_OBS:
        k, confidence = 1.0, "high"
    elif n >= BETA_SHRINK_FLOOR:
        # ramp from the 0.25 floor at BETA_SHRINK_FLOOR up to 1.0 at BETA_MIN_OBS
        t = (n - BETA_SHRINK_FLOOR) / (BETA_MIN_OBS - BETA_SHRINK_FLOOR)
        k = 0.25 + 0.75 * t
        confidence = "low"
    else:
        k, confidence = 0.25, "low"
    betas = {f: round(float(b * k), 4) for f, b in zip(FACTORS, beta_vec)}
    return Betas(ticker, betas, round(r2, 4), confidence, n)
