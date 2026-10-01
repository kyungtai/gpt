"""펌 오케스트레이터: 일간 운영 루프.

하루 순서 (t 종가 기준):
    1. 각 데스크 마킹 (t-1 포지션 -> t)            [손익 확정]
    2. 데스크 지표 갱신 (observe)
    3. 리스크 사후 점검: 데스크 손실/DD, 펌 디리스크/킬스위치
    4. (21영업일마다) CIO 배분기로 데스크 리스크 예산 재설정
    5. 유효 예산 = 기본예산 x 펌 스케일 x 데스크 스케일 x VaR 스케일
    6. 목표 포지션 산출 -> 그로스 한도 -> PaperBroker 체결 (비용 차감)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .allocator import AllocatorConfig, allocate
from .broker import make_broker
from .desks import Desk, default_desks
from .market import TRADING_DAYS, SyntheticMarket
from .risk import RiskLimits, RiskManager


@dataclass(frozen=True)
class FirmConfig:
    name: str = "Claude Capital (Paper)"
    capital: float = 100_000_000.0
    target_vol: float = 0.10  # 펌 연율 변동성 목표 (자본 대비)
    rebalance_every: int = 21
    broker_mode: str = "paper"
    allocator: AllocatorConfig = field(default_factory=AllocatorConfig)
    limits: RiskLimits = field(default_factory=RiskLimits)


@dataclass
class Results:
    firm: str
    dates: np.ndarray
    desk_names: list[str]
    equity: np.ndarray  # (T,)
    desk_pnl: np.ndarray  # (T, n) 순손익 (비용 차감 후)
    desk_cost: np.ndarray  # (T, n)
    annual_budget: np.ndarray  # (T, n) 각 일자 종가에 설정된 연간 $ 리스크 예산 (유효)
    gross: np.ndarray  # (T, n)
    firm_state: list[str]
    events: list
    allocations: list[dict]
    n_orders: int
    mm_fills: int
    capital: float


class Firm:
    def __init__(self, market: SyntheticMarket, cfg: FirmConfig | None = None, desks: list[Desk] | None = None):
        self.m = market
        self.cfg = cfg = cfg or FirmConfig()
        self.desks = desks if desks is not None else default_desks(seed=market.cfg.seed + 4)
        self.names = [d.name for d in self.desks]
        self.broker = make_broker(cfg.broker_mode)
        self.risk = RiskManager(self.names, cfg.limits)

    def run(self, progress: bool = False) -> Results:
        m, cfg, desks = self.m, self.cfg, self.desks
        T, n = m.n_days, len(desks)
        equity = np.full(T, cfg.capital)
        pnl = np.zeros((T, n))
        cost = np.zeros((T, n))
        norm = np.full((T, n), np.nan)
        eff_budget = np.zeros((T, n))
        gross = np.zeros((T, n))
        states = ["NORMAL"] * T
        allocations = []
        mm_fills = 0
        base = np.full(n, cfg.target_vol * cfg.capital / np.sqrt(n * (1 + (n - 1) * 0.1) / n) / n)
        eq = cfg.capital

        view0 = m.view(0)
        for d in desks:
            d.observe(view0)

        for t in range(1, T):
            view = m.view(t)
            date = view.date
            # 1) 마킹
            for i, d in enumerate(desks):
                r = d.mark(view)
                pnl[t, i] = r.pnl - r.cost
                cost[t, i] = r.cost
                mm_fills += r.fills
            eq += pnl[t].sum()
            # 2) 관측
            for d in desks:
                d.observe(view)
            # 3) 리스크 사후 점검
            prev_daily = eff_budget[t - 1] / np.sqrt(TRADING_DAYS)
            with np.errstate(invalid="ignore", divide="ignore"):
                norm[t] = np.where(prev_daily > 0, pnl[t] / prev_daily, np.nan)
            halted_now, _ = self.risk.post_mark(
                t, date,
                dict(zip(self.names, pnl[t])),
                dict(zip(self.names, prev_daily)),
                dict(zip(self.names, base)),
                eq,
            )
            # 4) 배분
            if t % cfg.rebalance_every == 0:
                active = np.array([d.ready(view) for d in desks])
                new_base, info = allocate(norm[: t + 1], active, cfg.target_vol * eq, cfg.allocator)
                if active.any():
                    base = np.where(active, new_base, base)
                    allocations.append({"t": t, "date": str(date), "budgets": base.copy(), **info})
            # 5) 유효 예산
            scales = np.array([self.risk.desk_scale(nm, t) for nm in self.names]) * self.risk.firm_scale()
            budgets = base * scales
            daily = budgets / np.sqrt(TRADING_DAYS)
            vs = self.risk.var_scale(t, date, norm[: t + 1], daily, eq)
            daily *= vs
            # 6) 목표 & 체결
            tgts = []
            for i, d in enumerate(desks):
                tgt = d.target(view, float(daily[i]))
                tgt = self.risk.gross_check(t, date, d, tgt, eq)
                tgt = self.risk.stress_check(t, date, d, tgt, view, eq)
                tgts.append(tgt)
            gsum = sum(d.gross_notional(tg) for d, tg in zip(desks, tgts))
            if gsum > cfg.limits.firm_gross_cap * eq:
                s = cfg.limits.firm_gross_cap * eq / gsum
                self.risk._log(t, date, "FIRM", "GROSS_CAP", f"펌 그로스 {gsum / eq:.1f}x; x{s:.2f}")
                tgts = [tg * s for tg in tgts]
            for i, d in enumerate(desks):
                c = self.broker.execute(t, d, tgts[i], view)
                pnl[t, i] -= c
                cost[t, i] += c
                eq -= c
                gross[t, i] = d.gross_notional()
            eff_budget[t] = daily * np.sqrt(TRADING_DAYS)
            equity[t] = eq
            states[t] = self.risk.firm_state
            if progress and t % 63 == 0:
                print(f"  {date} 자본 {eq / 1e6:,.2f}M  상태 {self.risk.firm_state}")

        return Results(
            firm=cfg.name,
            dates=m.dates,
            desk_names=self.names,
            equity=equity,
            desk_pnl=pnl,
            desk_cost=cost,
            annual_budget=eff_budget,
            gross=gross,
            firm_state=states,
            events=self.risk.events,
            allocations=allocations,
            n_orders=self.broker.n_orders,
            mm_fills=mm_fills,
            capital=cfg.capital,
        )

