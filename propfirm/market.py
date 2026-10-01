"""합성(synthetic) 멀티에셋 시장 시뮬레이터.

실제 시세가 아니다. 데스크/리스크/배분 로직을 운영 리허설하기 위한 데이터로,
각 자산군에 전략이 이용할 수 있는 '작고 현실적인' 엣지와 꼬리위험을 심어 둔다.

- 공통 레짐(평온/스트레스)과 주가 급락 점프
- 선물 13종 (주가지수/금리/FX/상품): 느리게 변하는 추세 드리프트 -> CTA 엣지
- 주가지수 내재변동성: 변동성 위험 프리미엄(VRP) + 노이즈 -> 변동성 데스크 엣지
- 지수 선물-현물 베이시스: OU 평균회귀 -> 델타원 엣지
- 국채 2s5s10s 커브: 곡률 팩터 평균회귀 -> RV 엣지
- 주식 30종목(5섹터): 잔차 OU 성분 -> 통계적 차익거래 엣지
- 장중 경로: 당일 수익률로 묶인 브라운 브리지 -> 마켓메이킹 시뮬레이션

모든 데이터는 생성 시점에 만들어지지만, 데스크는 `MarketView`를 통해
t 시점까지의 데이터만 볼 수 있다 (룩어헤드 방지).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252

# (심볼, 자산군, 연율 변동성)
FUTURES: list[tuple[str, str, float]] = [
    ("ES", "equity", 0.17),
    ("NQ", "equity", 0.22),
    ("FESX", "equity", 0.19),
    ("NKD", "equity", 0.21),
    ("ZN", "rates", 0.06),
    ("ZB", "rates", 0.11),
    ("FGBL", "rates", 0.06),
    ("6E", "fx", 0.08),
    ("6J", "fx", 0.10),
    ("6B", "fx", 0.09),
    ("CL", "commodity", 0.32),
    ("GC", "commodity", 0.15),
    ("HG", "commodity", 0.24),
]
FUT_SYMBOLS = [f[0] for f in FUTURES]
FUT_CLASS = [f[1] for f in FUTURES]
# 자산군별 (글로벌 팩터 로딩, 자산군 팩터 로딩)
CLASS_LOADINGS = {
    "equity": (0.7, 0.5),
    "rates": (-0.3, 0.85),
    "fx": (0.2, 0.7),
    "commodity": (0.3, 0.6),
}
VOL_UNDERLYINGS = ["ES", "NQ", "FESX"]
BASIS_UNDERLYINGS = ["ES", "FESX", "NKD"]
TENORS = ["2Y", "5Y", "10Y"]
N_SECTORS = 5
STOCKS_PER_SECTOR = 6


@dataclass(frozen=True)
class MarketConfig:
    n_days: int = 756
    seed: int = 7
    start_date: str = "2023-11-08"
    # 레짐
    p_calm_to_stress: float = 0.01
    p_stress_to_calm: float = 0.05
    stress_vol_mult: float = 2.0
    # 급락 점프 (글로벌 팩터 표준편차 단위)
    jump_prob_calm: float = 0.003
    jump_prob_stress: float = 0.02
    jump_mean: float = 4.0
    # 추세
    trend_persistence: float = 0.995
    trend_strength: float = 0.08  # 정상분포 드리프트 표준편차 (일간 변동성 단위)
    # 변동성
    vrp: float = 0.06  # 내재변동성 / 공정변동성 - 1
    iv_noise: float = 0.05
    # 베이시스
    basis_kappa: float = 0.06
    basis_sd: float = 0.0004
    # 금리 커브
    fly_kappa: float = 0.03
    fly_sd_bp: float = 1.5
    # 주식 잔차
    resid_kappa: float = 0.10
    resid_sd: float = 0.005
    idio_rw_sd: float = 0.007
    sector_sd: float = 0.007


class SyntheticMarket:
    def __init__(self, cfg: MarketConfig | None = None):
        self.cfg = cfg = cfg or MarketConfig()
        rng = np.random.default_rng(cfg.seed)
        T = cfg.n_days
        self.n_days = T
        self.dates = np.busday_offset(np.datetime64(cfg.start_date), np.arange(T), roll="forward")

        # --- 레짐 & 점프 -------------------------------------------------
        regime = np.zeros(T, dtype=int)
        for t in range(1, T):
            p = cfg.p_calm_to_stress if regime[t - 1] == 0 else 1 - cfg.p_stress_to_calm
            regime[t] = int(rng.random() < p)
        mult = np.where(regime == 1, cfg.stress_vol_mult, 1.0)
        jump_p = np.where(regime == 1, cfg.jump_prob_stress, cfg.jump_prob_calm)
        jumps = np.where(rng.random(T) < jump_p, -rng.normal(cfg.jump_mean, 1.0, T), 0.0)
        jumps[0] = 0.0
        self.regime, self.vol_mult, self.jumps = regime, mult, jumps

        # --- 선물 수익률 -------------------------------------------------
        F = len(FUTURES)
        ann_vol = np.array([f[2] for f in FUTURES])
        daily_vol = ann_vol / np.sqrt(TRADING_DAYS)
        classes = sorted(CLASS_LOADINGS)
        g = rng.standard_normal(T) + jumps
        cf = {c: rng.standard_normal(T) for c in classes}
        z = np.empty((T, F))
        for i, (_, cls, _) in enumerate(FUTURES):
            a, b = CLASS_LOADINGS[cls]
            z[:, i] = a * g + b * cf[cls] + np.sqrt(1 - a * a - b * b) * rng.standard_normal(T)
        phi = cfg.trend_persistence
        m = np.empty((T, F))
        m[0] = rng.normal(0, cfg.trend_strength, F)
        innov = cfg.trend_strength * np.sqrt(1 - phi * phi)
        for t in range(1, T):
            m[t] = phi * m[t - 1] + innov * rng.standard_normal(F)
        cond_vol = daily_vol[None, :] * mult[:, None]
        ret = cond_vol * (m + z)
        ret[0] = 0.0
        ret = np.maximum(ret, -0.5)
        self.fut_ret = ret
        self.fut_px = 100.0 * np.cumprod(1.0 + ret, axis=0)
        self.cond_vol = cond_vol
        self.fut_ann_vol = ann_vol

        # --- 내재변동성 (t 종가에 t+1 일을 위해 호가됨) -------------------
        P = np.array(
            [
                [1 - cfg.p_calm_to_stress, cfg.p_calm_to_stress],
                [cfg.p_stress_to_calm, 1 - cfg.p_stress_to_calm],
            ]
        )
        mult2 = np.array([1.0, cfg.stress_vol_mult**2])
        jp = np.array([cfg.jump_prob_calm, cfg.jump_prob_stress])
        a_eq = CLASS_LOADINGS["equity"][0]
        jump_var = a_eq**2 * jp * (cfg.jump_mean**2 + 1.0)
        fair_ratio = P @ (mult2 * (1.0 + jump_var))  # E[var_{t+1} | s_t] / base var
        vol_idx = [FUT_SYMBOLS.index(s) for s in VOL_UNDERLYINGS]
        noise = np.exp(cfg.iv_noise * rng.standard_normal((T, len(vol_idx))) - 0.5 * cfg.iv_noise**2)
        self.iv = (
            ann_vol[vol_idx][None, :]
            * np.sqrt(fair_ratio[regime])[:, None]
            * (1.0 + cfg.vrp)
            * noise
        )
        self.vol_idx = vol_idx

        # --- 지수 베이시스 (log F/F*) ------------------------------------
        B = len(BASIS_UNDERLYINGS)
        basis = np.zeros((T, B))
        for t in range(1, T):
            basis[t] = (1 - cfg.basis_kappa) * basis[t - 1] + cfg.basis_sd * mult[t] * rng.standard_normal(B)
        self.basis = basis

        # --- 금리 커브 ---------------------------------------------------
        level = 0.04 + np.cumsum(0.0006 * mult * rng.standard_normal(T))
        slope = np.zeros(T)
        slope[0] = 0.01
        curv = np.zeros(T)
        for t in range(1, T):
            slope[t] = 0.998 * slope[t - 1] + 0.002 * 0.01 + 0.0003 * rng.standard_normal()
            curv[t] = (1 - cfg.fly_kappa) * curv[t - 1] + cfg.fly_sd_bp * 1e-4 * mult[t] * rng.standard_normal()
        sl_load = np.array([-0.5, 0.0, 0.5])
        cv_load = np.array([-1 / 3, 2 / 3, -1 / 3])
        yields = (
            level[:, None]
            + slope[:, None] * sl_load[None, :]
            + curv[:, None] * cv_load[None, :]
            + 0.00004 * rng.standard_normal((T, 3))
        )
        self.yields = yields
        self.fly_bp = (2 * yields[:, 1] - yields[:, 0] - yields[:, 2]) * 1e4

        # --- 주식 유니버스 ------------------------------------------------
        N = N_SECTORS * STOCKS_PER_SECTOR
        self.sector = np.repeat(np.arange(N_SECTORS), STOCKS_PER_SECTOR)
        self.stock_names = [f"S{s}_{k}" for s in range(N_SECTORS) for k in range(STOCKS_PER_SECTOR)]
        mkt = np.cumsum(np.log1p(ret[:, FUT_SYMBOLS.index("ES")]))
        sec = np.cumsum(cfg.sector_sd * mult[:, None] * rng.standard_normal((T, N_SECTORS)), axis=0)
        w = np.cumsum(cfg.idio_rw_sd * mult[:, None] * rng.standard_normal((T, N)), axis=0)
        x = np.zeros((T, N))
        for t in range(1, T):
            x[t] = (1 - cfg.resid_kappa) * x[t - 1] + cfg.resid_sd * rng.standard_normal(N)
        logp = mkt[:, None] + sec[:, self.sector] + x + w
        logp -= logp[0]
        self.stock_px = 100.0 * np.exp(logp)

        for arr in (
            self.fut_ret,
            self.fut_px,
            self.cond_vol,
            self.iv,
            self.basis,
            self.yields,
            self.fly_bp,
            self.stock_px,
        ):
            arr.flags.writeable = False

    def view(self, t: int) -> "MarketView":
        return MarketView(self, t)


class MarketView:
    """t 시점 종가까지 관측 가능한 데이터만 노출한다."""

    def __init__(self, market: SyntheticMarket, t: int):
        if not 0 <= t < market.n_days:
            raise IndexError(t)
        self._m = market
        self.t = t

    @property
    def date(self):
        return self._m.dates[self.t]

    def _h(self, arr):
        return arr[: self.t + 1]

    @property
    def fut_px(self):
        return self._h(self._m.fut_px)

    @property
    def fut_ret(self):
        return self._h(self._m.fut_ret)

    @property
    def iv(self):
        return self._h(self._m.iv)

    @property
    def basis(self):
        return self._h(self._m.basis)

    @property
    def yields(self):
        return self._h(self._m.yields)

    @property
    def fly_bp(self):
        return self._h(self._m.fly_bp)

    @property
    def stock_px(self):
        return self._h(self._m.stock_px)

    @property
    def sector(self):
        return self._m.sector

    @property
    def vol_idx(self):
        return self._m.vol_idx

    def intraday_path(self, fut_idx: int, steps: int) -> np.ndarray:
        """당일(t) 장중 누적수익률 경로 (길이 steps+1, 0에서 시작해 당일 수익률로 끝남).

        시장이 생성하는 경로이며, 데스크는 이 경로를 단계별로 순서대로 소비해야 한다.
        """
        m = self._m
        rng = np.random.default_rng([m.cfg.seed, self.t, fut_idx, 99])
        sd = m.cond_vol[self.t, fut_idx] / np.sqrt(steps)
        w = np.concatenate([[0.0], np.cumsum(sd * rng.standard_normal(steps))])
        frac = np.arange(steps + 1) / steps
        return w - frac * (w[-1] - m.fut_ret[self.t, fut_idx])
