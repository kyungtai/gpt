"""트레이딩 데스크 (전략) 구현.

각 데스크의 하루 수명주기 (Firm이 호출):
    1. mark(view)    : t-1 포지션을 t 시점으로 평가 -> MarkResult
    2. observe(view) : t 시점 데이터로 내부 지표 갱신 (중단된 데스크도 호출됨)
    3. target(view, daily_budget) : t 종가 기준 목표 포지션 (일간 $ 리스크 예산)
    4. trade_cost(old, new, view) : 체결 비용 ($, PaperBroker가 호출)

포지션 단위는 데스크마다 다르며 (명목 $, 분산 명목, $/bp 등), 모두 선형으로 스케일 가능하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import (
    BASIS_UNDERLYINGS,
    FUT_CLASS,
    FUT_SYMBOLS,
    TRADING_DAYS,
    VOL_UNDERLYINGS,
    MarketView,
)


def _alpha(halflife: float) -> float:
    return 1.0 - 0.5 ** (1.0 / halflife)


class EWMean:
    def __init__(self, halflife: float, shape=()):
        self.a = _alpha(halflife)
        self.value = np.zeros(shape)
        self.n = 0

    def update(self, x):
        # 초기에는 단순평균 (콜드스타트 편향 제거), 이후 지수가중
        x = np.asarray(x, dtype=float)
        self.n += 1
        a = max(self.a, 1.0 / self.n)
        self.value = self.value + a * (x - self.value)
        return self.value


class EWCov:
    """평균 0 가정 지수가중 공분산 (일간 수익률용)."""

    def __init__(self, halflife: float, n: int):
        self.a = _alpha(halflife)
        self.value = np.zeros((n, n))
        self.n = 0

    def update(self, x):
        x = np.asarray(x, dtype=float)
        outer = np.outer(x, x)
        self.n += 1
        a = max(self.a, 1.0 / self.n)
        self.value = (1 - a) * self.value + a * outer
        return self.value

    @property
    def vol(self):
        return np.sqrt(np.maximum(np.diag(self.value), 1e-12))


@dataclass
class MarkResult:
    pnl: float  # 비용 차감 전 손익
    cost: float = 0.0  # 장중 수수료 등 (마켓메이킹)
    fills: int = 0


class Desk:
    name = "BASE"
    strategy = ""
    instruments: list[str] = []
    warmup = 0

    def __init__(self):
        self.pos = np.zeros(len(self.instruments))

    def mark(self, view: MarketView) -> MarkResult:
        raise NotImplementedError

    def observe(self, view: MarketView) -> None:
        pass

    def target(self, view: MarketView, daily_budget: float) -> np.ndarray:
        raise NotImplementedError

    def trade_cost(self, old: np.ndarray, new: np.ndarray, view: MarketView) -> float:
        raise NotImplementedError

    def gross_notional(self, pos: np.ndarray | None = None) -> float:
        p = self.pos if pos is None else pos
        return float(np.abs(p).sum())

    def ready(self, view: MarketView) -> bool:
        return view.t >= self.warmup

    def stress_loss(self, pos: np.ndarray, view: MarketView) -> float:
        """사전 스트레스 시나리오 손실 ($, 양수=손실). 선형 데스크는 VaR/그로스 한도로 관리하므로 0."""
        return 0.0


def _scale_to_budget(w: np.ndarray, cov: np.ndarray, daily_budget: float) -> np.ndarray:
    var = float(w @ cov @ w)
    if var <= 0 or not np.isfinite(var):
        return np.zeros_like(w)
    return w * (daily_budget / np.sqrt(var))


def _no_trade_band(old: np.ndarray, new: np.ndarray, frac: float) -> np.ndarray:
    keep = np.abs(new - old) < frac * np.abs(new)
    return np.where(keep, old, new)


# ---------------------------------------------------------------------------
# CTA: 멀티 자산 추세추종
# ---------------------------------------------------------------------------
class CTADesk(Desk):
    name = "CTA"
    strategy = "추세추종 (EWMA 크로스오버 3종, 변동성 타게팅)"
    instruments = FUT_SYMBOLS
    warmup = 130
    spans = [(8, 32), (16, 64), (32, 128)]
    cost_bp = {"equity": 0.6, "rates": 0.4, "fx": 0.5, "commodity": 1.5}

    def __init__(self):
        super().__init__()
        n = len(self.instruments)
        self.fast = [EWMean(f / 1.44, n) for f, _ in self.spans]
        self.slow = [EWMean(s / 1.44, n) for _, s in self.spans]
        self.sig_scale = [EWMean(250, n) for _ in self.spans]
        self.cov = EWCov(60, n)
        self.bp = np.array([self.cost_bp[c] for c in FUT_CLASS])
        self._fc = np.zeros(n)

    def mark(self, view):
        r = view.fut_ret[-1]
        pnl = float(self.pos @ r)
        self.pos = self.pos * (1.0 + r)
        return MarkResult(pnl)

    def observe(self, view):
        r = view.fut_ret[-1]
        lp = np.log(view.fut_px[-1])
        if view.t == 0:
            for k in range(len(self.spans)):
                self.fast[k].update(lp)
                self.slow[k].update(lp)
            return
        self.cov.update(r)
        vol = self.cov.vol
        fc = np.zeros_like(vol)
        for k in range(len(self.spans)):
            self.fast[k].update(lp)
            self.slow[k].update(lp)
            raw = (self.fast[k].value - self.slow[k].value) / vol
            scale = np.sqrt(self.sig_scale[k].update(raw * raw))
            fc += np.clip(raw / np.maximum(scale, 1e-9), -2.0, 2.0)
        self._fc = fc / len(self.spans)

    def target(self, view, daily_budget):
        if not self.ready(view) or daily_budget <= 0:
            return np.zeros_like(self.pos)
        w = self._fc / self.cov.vol
        tgt = _scale_to_budget(w, self.cov.value, daily_budget)
        return _no_trade_band(self.pos, tgt, 0.10)

    def trade_cost(self, old, new, view):
        return float(np.abs(new - old) @ self.bp) * 1e-4


# ---------------------------------------------------------------------------
# 통계적 차익거래: 섹터 중립 잔차 평균회귀
# ---------------------------------------------------------------------------
class StatArbDesk(Desk):
    name = "STATARB"
    strategy = "섹터 중립 잔차 평균회귀 (30종목, 달러 중립)"
    warmup = 80
    cost_bp = 3.0

    def __init__(self, n_stocks: int = 30):
        self.instruments = [f"S{i}" for i in range(n_stocks)]
        super().__init__()
        self.mean = EWMean(10, n_stocks)
        self.var = EWMean(60, n_stocks)
        self.cov = EWCov(60, n_stocks)
        self._z = np.zeros(n_stocks)

    def mark(self, view):
        px = view.stock_px
        r = px[-1] / px[-2] - 1.0
        pnl = float(self.pos @ r)
        self.pos = self.pos * (1.0 + r)
        return MarkResult(pnl)

    def observe(self, view):
        px = view.stock_px
        sec = view.sector
        lp = np.log(px[-1])
        sec_mean = np.bincount(sec, lp) / np.bincount(sec)
        resid = lp - sec_mean[sec]
        if self.mean.n:
            dev = resid - self.mean.value
            self.var.update(dev * dev)
            self._z = dev / np.sqrt(np.maximum(self.var.value, 1e-10))
        self.mean.update(resid)
        if len(px) >= 2:
            self.cov.update(px[-1] / px[-2] - 1.0)

    def target(self, view, daily_budget):
        if not self.ready(view) or daily_budget <= 0:
            return np.zeros_like(self.pos)
        sec = view.sector
        w = -np.clip(self._z, -3, 3)
        w = w - (np.bincount(sec, w) / np.bincount(sec))[sec]
        tgt = _scale_to_budget(w, self.cov.value, daily_budget)
        return _no_trade_band(self.pos, tgt, 0.25)

    def trade_cost(self, old, new, view):
        return float(np.abs(new - old).sum()) * self.cost_bp * 1e-4


# ---------------------------------------------------------------------------
# 델타원: 지수 선물-현물 베이시스 수렴
# ---------------------------------------------------------------------------
class DeltaOneDesk(Desk):
    name = "DELTA1"
    strategy = "지수 선물/현물 베이시스 평균회귀 (양다리 헤지)"
    instruments = [f"{s}_basis" for s in BASIS_UNDERLYINGS]
    warmup = 60
    cost_bp = 1.5  # 양다리 합산
    entry_z, exit_z = 0.75, 0.25

    def __init__(self):
        super().__init__()
        n = len(self.instruments)
        self.mean = EWMean(60, n)
        self.var = EWMean(120, n)
        self.dvar = EWMean(60, n)
        self._z = np.zeros(n)

    def mark(self, view):
        db = view.basis[-1] - view.basis[-2]
        return MarkResult(float(self.pos @ db))

    def observe(self, view):
        b = view.basis[-1]
        if self.mean.n:
            dev = b - self.mean.value
            self.var.update(dev * dev)
            self._z = dev / np.sqrt(np.maximum(self.var.value, 1e-14))
            self.dvar.update((b - view.basis[-2]) ** 2)
        self.mean.update(b)

    def target(self, view, daily_budget):
        if not self.ready(view) or daily_budget <= 0:
            return np.zeros_like(self.pos)
        z = self._z
        unit = daily_budget / (np.sqrt(np.maximum(self.dvar.value, 1e-14)) * np.sqrt(len(z)))
        sig = -np.clip(z, -2.5, 2.5) / 2.5
        holding = (np.sign(self.pos) == np.sign(sig)) & (self.pos != 0)
        active = (np.abs(z) > self.entry_z) | (holding & (np.abs(z) > self.exit_z))
        return np.where(active, sig * unit, 0.0)

    def trade_cost(self, old, new, view):
        return float(np.abs(new - old).sum()) * self.cost_bp * 1e-4

    def gross_notional(self, pos=None):
        return 2.0 * super().gross_notional(pos)


# ---------------------------------------------------------------------------
# RV: 국채 2s5s10s 버터플라이
# ---------------------------------------------------------------------------
class RatesRVDesk(Desk):
    name = "RATES_RV"
    strategy = "2s5s10s 버터플라이 평균회귀 (DV01 중립)"
    instruments = ["2s5s10s_fly"]  # 포지션 단위: $/bp, 양수 = 플라이 금리 하락에 베팅
    warmup = 60
    cost_fly_bp = 0.4
    dv01_to_notional = 1.0 / 0.00045 * 2  # 대략적 그로스 명목 환산 (5Y 듀레이션 ~4.5, 3 legs)

    def __init__(self):
        super().__init__()
        self.mean = EWMean(60)
        self.var = EWMean(120)
        self.dvar = EWMean(60)
        self._z = 0.0

    def mark(self, view):
        d = view.fly_bp[-1] - view.fly_bp[-2]
        return MarkResult(float(-self.pos[0] * d))

    def observe(self, view):
        f = view.fly_bp[-1]
        if self.mean.n:
            dev = f - float(self.mean.value)
            self.var.update(dev * dev)
            self._z = dev / np.sqrt(max(float(self.var.value), 1e-8))
            self.dvar.update((f - view.fly_bp[-2]) ** 2)
        self.mean.update(f)

    def target(self, view, daily_budget):
        if not self.ready(view) or daily_budget <= 0 or abs(self._z) < 0.5:
            return np.zeros(1)
        unit = 1.5 * daily_budget / np.sqrt(max(float(self.dvar.value), 1e-8))
        tgt = np.array([np.clip(self._z, -2.5, 2.5) / 2.5 * unit])
        return _no_trade_band(self.pos, tgt, 0.15)

    def trade_cost(self, old, new, view):
        return float(np.abs(new - old).sum()) * self.cost_fly_bp

    def gross_notional(self, pos=None):
        return super().gross_notional(pos) * self.dv01_to_notional


# ---------------------------------------------------------------------------
# 변동성: 델타헤지 단기 분산 매도/매수
# ---------------------------------------------------------------------------
class VolDesk(Desk):
    name = "VOL"
    strategy = "내재 vs 예측 변동성 (델타헤지 분산, 스트레스 필터)"
    instruments = [f"{s}_var" for s in VOL_UNDERLYINGS]  # 양수 = 분산 매도
    warmup = 40
    cost_frac = 0.03
    # 단위 분산명목당 일간 손익 표준편차 × 3종목 상관(≈0.8) 합산
    unit_risk = 1.2 * np.sqrt(3 + 6 * 0.8)

    def __init__(self):
        super().__init__()
        n = len(self.instruments)
        self.var_long = EWMean(20, n)
        self.var_short = EWMean(5, n)

    def mark(self, view):
        if not self.pos.any():
            return MarkResult(0.0)
        r = view.fut_ret[-1, view.vol_idx]
        iv_prev = view.iv[-2]
        implied_daily_var = iv_prev**2 / TRADING_DAYS
        return MarkResult(float(self.pos @ (1.0 - r * r / implied_daily_var)))

    def observe(self, view):
        if view.t == 0:
            return
        r = view.fut_ret[-1, view.vol_idx]
        self.var_long.update(r * r)
        self.var_short.update(r * r)

    def target(self, view, daily_budget):
        if not self.ready(view) or daily_budget <= 0:
            return np.zeros_like(self.pos)
        fcst = np.sqrt(self.var_long.value * TRADING_DAYS)
        ratio = view.iv[-1] / np.maximum(fcst, 1e-6)
        stress = np.sqrt(self.var_short.value / np.maximum(self.var_long.value, 1e-12)) > 1.5
        sig = np.where(ratio > 1.05, np.minimum((ratio - 1.0) / 0.15, 1.0), 0.0)
        sig = np.where(stress, 0.0, sig)
        sig = np.where(ratio < 0.92, -0.5 * np.minimum((1.0 - ratio) / 0.15, 1.0), sig)
        tgt = sig * daily_budget / self.unit_risk
        return _no_trade_band(self.pos, tgt, 0.20)

    def trade_cost(self, old, new, view):
        return float(np.abs(new - old).sum()) * self.cost_frac

    stress_move = 0.10  # 기초자산 동시 10% 갭

    def stress_loss(self, pos, view):
        implied_daily_var = view.iv[-1] ** 2 / TRADING_DAYS
        pnl = pos @ (1.0 - self.stress_move**2 / implied_daily_var)
        return float(max(-pnl, 0.0))


# ---------------------------------------------------------------------------
# 마켓메이킹: Avellaneda-Stoikov 호가, 장중 시뮬레이션, 오버나이트 플랫
# ---------------------------------------------------------------------------
class MarketMakingDesk(Desk):
    name = "MM"
    strategy = "Avellaneda-Stoikov 양방향 호가 (재고 스큐, 정보거래자 역선택 반영)"
    instruments = ["ES", "6E", "ZN", "CL"]
    warmup = 40
    steps = 78  # 5분봉
    gamma = 0.1  # 위험회피 (σ 단위)
    kappa = 1.5  # 체결강도 감쇠
    A = 0.3  # 단계당 기본 체결강도
    horizon = 10
    q_max = 6
    p_informed = 0.5
    close_cost = 0.3  # 장마감 재고 청산 비용 (σ_step 단위)
    fee_ret = 0.00002  # 체결당 거래소 수수료 (명목 대비)
    risk_per_lot = 1.25  # 로트당 일간 손익 표준편차 (σ_daily 단위, 보수적 추정)

    def __init__(self, seed: int = 11):
        super().__init__()
        self.idx = [FUT_SYMBOLS.index(s) for s in self.instruments]
        self.var = EWMean(20, len(self.instruments))
        self.rng = np.random.default_rng(seed)
        self.lot = np.zeros(len(self.instruments))  # 오늘 사용할 로트 명목 ($)
        self.last_inventory_peak = 0

    def mark(self, view):
        if not self.lot.any():
            return MarkResult(0.0)
        n = len(self.idx)
        sig_d = np.sqrt(np.maximum(self.var.value, 1e-10))  # 어제까지의 추정치
        sig_s = sig_d / np.sqrt(self.steps)
        x = np.stack([view.intraday_path(i, self.steps) for i in self.idx]) / sig_s[:, None]
        q = np.zeros(n)
        cash = np.zeros(n)
        fills = 0
        peak = 0.0
        base_half = np.log(1 + self.gamma / self.kappa) / self.gamma
        for k in range(self.steps):
            s = x[:, k]
            dx = x[:, k + 1] - s
            tau = min(self.steps - k, self.horizon)
            res = s - q * self.gamma * tau
            half = base_half + self.gamma * tau / 2
            ask, bid = res + half, res - half
            da, db = np.maximum(ask - s, 0.0), np.maximum(s - bid, 0.0)
            u = self.rng.random((4, n))
            hit_ask = (u[0] < self.A * np.exp(-self.kappa * da)) | ((dx > da) & (u[1] < self.p_informed))
            hit_bid = (u[2] < self.A * np.exp(-self.kappa * db)) | ((dx < -db) & (u[3] < self.p_informed))
            hit_ask &= q > -self.q_max
            hit_bid &= q < self.q_max
            q = q - hit_ask + hit_bid
            cash = cash + hit_ask * ask - hit_bid * bid
            fills += int(hit_ask.sum() + hit_bid.sum())
            peak = max(peak, float(np.abs(q).max()))
        pnl_sigma = cash + q * x[:, -1] - np.abs(q) * self.close_cost
        pnl = float((pnl_sigma * sig_s * self.lot).sum())
        fee = float(fills * self.fee_ret * self.lot.mean())
        self.last_inventory_peak = peak
        return MarkResult(pnl, cost=fee, fills=fills)

    def observe(self, view):
        if view.t == 0:
            return
        r = view.fut_ret[-1, self.idx]
        self.var.update(r * r)

    def target(self, view, daily_budget):
        # 오버나이트 포지션 없음. 다음 거래일 로트 크기만 정한다.
        if not self.ready(view) or daily_budget <= 0:
            self.lot = np.zeros(len(self.idx))
        else:
            sig_d = np.sqrt(np.maximum(self.var.value, 1e-10))
            self.lot = daily_budget / (np.sqrt(len(self.idx)) * self.risk_per_lot * sig_d)
        return np.zeros_like(self.pos)

    def trade_cost(self, old, new, view):
        return 0.0

    def gross_notional(self, pos=None):
        return float(self.lot.sum() * self.q_max)


def default_desks(seed: int = 11) -> list[Desk]:
    return [
        DeltaOneDesk(),
        MarketMakingDesk(seed=seed),
        StatArbDesk(),
        VolDesk(),
        RatesRVDesk(),
        CTADesk(),
    ]
