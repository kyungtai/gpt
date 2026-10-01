"""리스크 관리: 데스크 손실한도, 데스크 드로다운 정지, 펌 디리스크/킬스위치, VaR, 그로스 한도."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RiskLimits:
    desk_daily_loss_mult: float = 4.0  # 일간 손실 > 4 x 일간 예산 -> 플랫 + 5일 정지
    desk_daily_loss_halt_days: int = 5
    desk_dd_mult: float = 1.25  # 데스크 DD > 1.25 x 연간 예산 -> 21일 정지
    desk_dd_halt_days: int = 21
    probation_days: int = 63
    probation_scale: float = 0.5
    firm_derisk_dd: float = 0.06
    firm_kill_dd: float = 0.10
    firm_restore_dd: float = 0.03
    firm_kill_days: int = 10
    firm_recovery_days: int = 21
    firm_recovery_scale: float = 0.25
    var_limit: float = 0.02  # 99% 1일 VaR <= 자본의 2%
    var_lookback: int = 250
    gross_caps: dict = field(
        default_factory=lambda: {"CTA": 3.0, "STATARB": 4.0, "DELTA1": 12.0, "RATES_RV": 10.0}
    )
    firm_gross_cap: float = 20.0
    desk_stress_limit: float = 0.015  # 데스크 스트레스 손실 <= 자본의 1.5%


@dataclass
class Event:
    t: int
    date: str
    scope: str
    kind: str
    detail: str


@dataclass
class DeskRiskState:
    halted_until: int = -1
    probation_until: int = -1
    cum: float = 0.0
    hwm: float = 0.0

    def halted(self, t):
        return t < self.halted_until

    def scale(self, t, limits: RiskLimits):
        if self.halted(t):
            return 0.0
        return limits.probation_scale if t < self.probation_until else 1.0


class RiskManager:
    def __init__(self, desk_names: list[str], limits: RiskLimits = RiskLimits()):
        self.limits = limits
        self.names = desk_names
        self.desks = {n: DeskRiskState() for n in desk_names}
        self.events: list[Event] = []
        self.firm_state = "NORMAL"
        self.firm_hwm = None
        self.firm_until = -1

    def _log(self, t, date, scope, kind, detail):
        self.events.append(Event(t, str(date), scope, kind, detail))

    # -- 사후 (마킹 후) 점검 ------------------------------------------------
    def post_mark(self, t, date, desk_pnl: dict, daily_budget_prev: dict, annual_budget: dict, equity: float):
        L = self.limits
        newly_halted = []
        for name, pnl in desk_pnl.items():
            st = self.desks[name]
            st.cum += pnl
            st.hwm = max(st.hwm, st.cum)
            if st.halted(t):
                continue
            b = daily_budget_prev.get(name, 0.0)
            if b > 0 and pnl < -L.desk_daily_loss_mult * b:
                st.halted_until = t + L.desk_daily_loss_halt_days
                newly_halted.append(name)
                self._log(t, date, name, "DAILY_LOSS_LIMIT",
                          f"손실 {pnl:,.0f} < -{L.desk_daily_loss_mult}x 일간예산({b:,.0f}); 플랫 후 {L.desk_daily_loss_halt_days}일 정지")
                continue
            ab = annual_budget.get(name, 0.0)
            dd = st.hwm - st.cum
            if ab > 0 and dd > L.desk_dd_mult * ab:
                st.halted_until = t + L.desk_dd_halt_days
                st.probation_until = st.halted_until + L.probation_days
                st.hwm = st.cum
                newly_halted.append(name)
                self._log(t, date, name, "DESK_DRAWDOWN",
                          f"DD {dd:,.0f} > {L.desk_dd_mult}x 연간예산({ab:,.0f}); {L.desk_dd_halt_days}일 정지 후 "
                          f"{L.probation_days}일 {L.probation_scale:.0%} 근신")

        # 펌 레벨 상태머신
        if self.firm_hwm is None:
            self.firm_hwm = equity
        self.firm_hwm = max(self.firm_hwm, equity)
        dd = 1.0 - equity / self.firm_hwm
        s = self.firm_state
        if s in ("NORMAL", "DERISK", "RECOVERY") and dd >= L.firm_kill_dd:
            self.firm_state, self.firm_until = "KILLED", t + L.firm_kill_days
            self._log(t, date, "FIRM", "KILL_SWITCH", f"펌 DD {dd:.1%} >= {L.firm_kill_dd:.0%}; 전 데스크 플랫, {L.firm_kill_days}일 정지")
        elif s == "NORMAL" and dd >= L.firm_derisk_dd:
            self.firm_state = "DERISK"
            self._log(t, date, "FIRM", "DERISK", f"펌 DD {dd:.1%} >= {L.firm_derisk_dd:.0%}; 리스크 50% 축소")
        elif s == "DERISK" and dd < L.firm_restore_dd:
            self.firm_state = "NORMAL"
            self._log(t, date, "FIRM", "RESTORE", f"펌 DD {dd:.1%} < {L.firm_restore_dd:.0%}; 정상 리스크 복귀")
        elif s == "KILLED" and t >= self.firm_until:
            self.firm_state, self.firm_until = "RECOVERY", t + L.firm_recovery_days
            self.firm_hwm = equity  # 사후 리뷰 완료 -> 기준점 재설정
            self._log(t, date, "FIRM", "RESUME", f"리뷰 후 재개; {L.firm_recovery_days}일간 {L.firm_recovery_scale:.0%} 리스크")
        elif s == "RECOVERY" and t >= self.firm_until:
            self.firm_state = "DERISK"
            self._log(t, date, "FIRM", "RECOVERY_DONE", "회복기간 종료; 50% 리스크(디리스크)로 전환")
        return newly_halted, dd

    def firm_scale(self) -> float:
        return {"NORMAL": 1.0, "DERISK": 0.5, "KILLED": 0.0, "RECOVERY": self.limits.firm_recovery_scale}[self.firm_state]

    def desk_scale(self, name, t) -> float:
        return self.desks[name].scale(t, self.limits)

    # -- 사전 (주문 전) 점검 -------------------------------------------------
    def var_scale(self, t, date, norm_ret: np.ndarray, daily_budgets: np.ndarray, equity: float) -> float:
        L = self.limits
        hist = norm_ret[-L.var_lookback :]
        if len(hist) < 60 or not daily_budgets.any():
            return 1.0
        sim = np.nan_to_num(hist) @ daily_budgets
        var99 = -np.quantile(sim, 0.01)
        limit = L.var_limit * equity
        if var99 > limit:
            s = limit / var99
            self._log(t, date, "FIRM", "VAR_SCALE", f"사전 VaR99 {var99:,.0f} > 한도 {limit:,.0f}; 전 데스크 x{s:.2f}")
            return s
        return 1.0

    def gross_check(self, t, date, desk, tgt: np.ndarray, equity: float) -> np.ndarray:
        cap = self.limits.gross_caps.get(desk.name)
        if cap is None:
            return tgt
        g = desk.gross_notional(tgt)
        if g > cap * equity:
            s = cap * equity / g
            self._log(t, date, desk.name, "GROSS_CAP", f"그로스 {g / equity:.1f}x > {cap}x; x{s:.2f}")
            return tgt * s
        return tgt

    def stress_check(self, t, date, desk, tgt: np.ndarray, view, equity: float) -> np.ndarray:
        loss = desk.stress_loss(tgt, view)
        limit = self.limits.desk_stress_limit * equity
        if loss > limit:
            s = limit / loss
            self._log(t, date, desk.name, "STRESS_LIMIT", f"스트레스 손실 {loss:,.0f} > 한도 {limit:,.0f}; x{s:.2f}")
            return tgt * s
        return tgt
