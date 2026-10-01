"""체결 계층. 페이퍼(모의) 브로커만 존재한다.

실주문 연결은 의도적으로 구현하지 않았다. AGENTS.md 규칙에 따라 실제 주문은
사용자의 명시적인 별도 지시와 별도 구현/검토가 있어야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class LiveTradingDisabled(RuntimeError):
    pass


@dataclass
class Fill:
    t: int
    desk: str
    instrument: str
    old: float
    new: float


@dataclass
class PaperBroker:
    mode: str = "paper"
    record_fills: bool = False
    fills: list[Fill] = field(default_factory=list)
    n_orders: int = 0

    def execute(self, t, desk, new_pos: np.ndarray, view) -> float:
        """desk.pos -> new_pos로 리밸런스하고 비용($)을 반환한다."""
        old = desk.pos
        cost = desk.trade_cost(old, new_pos, view)
        changed = np.flatnonzero(~np.isclose(old, new_pos, rtol=0, atol=1e-9))
        self.n_orders += len(changed)
        if self.record_fills:
            for i in changed:
                self.fills.append(Fill(t, desk.name, desk.instruments[i], float(old[i]), float(new_pos[i])))
        desk.pos = np.asarray(new_pos, dtype=float).copy()
        return cost


def make_broker(mode: str = "paper", **kwargs) -> PaperBroker:
    if mode != "paper":
        raise LiveTradingDisabled(
            f"'{mode}' 모드는 비활성화되어 있습니다. 실주문은 사용자의 명시적 별도 지시와 "
            "별도 구현·리스크 검토 후에만 추가합니다."
        )
    return PaperBroker(**kwargs)
