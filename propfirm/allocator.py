"""CIO 자본 배분기: 성과 기울기 + 상관 인식 리스크 버짓.

데스크 손익을 '예산 단위 수익률'(일간 손익 / 일간 리스크 예산)로 정규화한 뒤
    1) 최근 126일 샤프로 지수 가중 (온라인 학습형 성과 기울기)
    2) 데스크별 최소/최대 비중 제약
    3) 수축(shrinkage) 상관행렬로 사전 펌 변동성이 목표가 되도록 스케일
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import TRADING_DAYS


@dataclass(frozen=True)
class AllocatorConfig:
    lookback: int = 126
    min_history: int = 63
    eta: float = 0.35  # 샤프 기울기 강도
    w_min: float = 0.05
    w_max: float = 0.35
    corr_shrink: float = 0.5
    prior_corr: float = 0.1
    max_budget_sum_mult: float = 3.0  # 데스크 예산 합 <= 3 x 펌 예산


def _cap_weights(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """raw에 비례하되 [lo, hi] 경계를 만족하고 합이 1인 비중 (워터필링)."""
    n = len(raw)
    w = np.zeros(n)
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        free = ~fixed
        mass = 1.0 - w[fixed].sum()
        w[free] = raw[free] / raw[free].sum() * mass
        over = free & (w > hi)
        under = free & (w < lo)
        if not over.any() and not under.any():
            break
        w[over], w[under] = hi, lo
        fixed |= over | under
    return w


def allocate(
    norm_ret: np.ndarray,
    active: np.ndarray,
    firm_annual_budget: float,
    cfg: AllocatorConfig = AllocatorConfig(),
) -> tuple[np.ndarray, dict]:
    """norm_ret: (T, n) 예산 단위 일간 수익률 (NaN = 미운용). 반환: 데스크별 연간 $ 리스크 예산."""
    n = norm_ret.shape[1]
    budgets = np.zeros(n)
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return budgets, {"weights": budgets, "sharpe": np.full(n, np.nan), "div": np.nan}

    window = norm_ret[-cfg.lookback :, idx]
    counts = np.isfinite(window).sum(axis=0)
    sharpe = np.zeros(len(idx))
    for j in range(len(idx)):
        x = window[np.isfinite(window[:, j]), j]
        if counts[j] >= cfg.min_history and x.std() > 0:
            sharpe[j] = x.mean() / x.std() * np.sqrt(TRADING_DAYS)
    raw = np.exp(cfg.eta * np.clip(sharpe, -2.0, 3.0))
    lo = min(cfg.w_min, 1.0 / len(idx))
    hi = max(cfg.w_max, 1.0 / len(idx))
    w = _cap_weights(raw, lo, hi)

    corr = np.full((len(idx), len(idx)), cfg.prior_corr)
    np.fill_diagonal(corr, 1.0)
    rows = np.isfinite(window).all(axis=1)
    if rows.sum() >= cfg.min_history and len(idx) > 1:
        x = window[rows]
        sd = x.std(axis=0)
        ok = sd > 0
        if ok.sum() > 1:
            sample = corr.copy()
            sample[np.ix_(ok, ok)] = np.corrcoef(x[:, ok].T)
            corr = cfg.corr_shrink * corr + (1 - cfg.corr_shrink) * sample
    div = float(np.sqrt(w @ corr @ w))
    b = w * firm_annual_budget / div
    if b.sum() > cfg.max_budget_sum_mult * firm_annual_budget:
        b *= cfg.max_budget_sum_mult * firm_annual_budget / b.sum()
    budgets[idx] = b
    full_w = np.zeros(n)
    full_w[idx] = w
    full_sr = np.full(n, np.nan)
    full_sr[idx] = sharpe
    return budgets, {"weights": full_w, "sharpe": full_sr, "div": div}
