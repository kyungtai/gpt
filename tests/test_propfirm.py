import numpy as np
import pytest

from propfirm import Firm, FirmConfig, MarketConfig, SyntheticMarket
from propfirm.allocator import allocate
from propfirm.broker import LiveTradingDisabled, make_broker
from propfirm.desks import CTADesk, MarketMakingDesk, VolDesk
from propfirm.report import summarize, to_markdown
from propfirm.risk import RiskLimits, RiskManager


@pytest.fixture(scope="module")
def small_run():
    m = SyntheticMarket(MarketConfig(n_days=300, seed=3))
    return m, Firm(m).run()


def test_market_is_deterministic():
    a = SyntheticMarket(MarketConfig(n_days=50, seed=1))
    b = SyntheticMarket(MarketConfig(n_days=50, seed=1))
    assert np.array_equal(a.fut_px, b.fut_px)
    assert np.array_equal(a.view(10).intraday_path(0, 10), b.view(10).intraday_path(0, 10))


def test_view_has_no_lookahead():
    m = SyntheticMarket(MarketConfig(n_days=50, seed=1))
    v = m.view(20)
    for arr in (v.fut_px, v.fut_ret, v.iv, v.basis, v.yields, v.fly_bp, v.stock_px):
        assert len(arr) == 21
    with pytest.raises(ValueError):
        v.fut_px[0, 0] = 1.0  # 시장 데이터는 읽기 전용


def test_intraday_path_matches_daily_return():
    m = SyntheticMarket(MarketConfig(n_days=30, seed=2))
    p = m.view(5).intraday_path(3, 78)
    assert p[0] == 0.0
    assert np.isclose(p[-1], m.fut_ret[5, 3])


def test_live_broker_is_disabled():
    with pytest.raises(LiveTradingDisabled):
        make_broker("live")
    assert make_broker("paper").mode == "paper"
    with pytest.raises(LiveTradingDisabled):
        Firm(SyntheticMarket(MarketConfig(n_days=10)), FirmConfig(broker_mode="live"))


def test_allocator_respects_caps_and_targets_vol():
    rng = np.random.default_rng(0)
    norm = rng.standard_normal((200, 4)) + np.array([0.2, 0.0, -0.1, 0.05])
    active = np.array([True, True, True, False])
    b, info = allocate(norm, active, 10e6)
    assert b[3] == 0
    w = info["weights"][:3]
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0.05 - 1e-9).all() and (w <= 0.35 + 1e-9).all()
    assert b[0] > b[2]  # 성과 좋은 데스크가 더 많은 예산
    # 독립 데스크 -> 예산 합이 펌 예산보다 크다 (분산효과)
    assert b.sum() > 10e6


def test_firm_kill_switch_and_recovery():
    rm = RiskManager(["A"], RiskLimits())
    eq = 100.0
    rm.post_mark(1, "d1", {"A": 0.0}, {"A": 1.0}, {"A": 10.0}, eq)
    rm.post_mark(2, "d2", {"A": 0.0}, {"A": 1.0}, {"A": 10.0}, 93.0)
    assert rm.firm_state == "DERISK" and rm.firm_scale() == 0.5
    rm.post_mark(3, "d3", {"A": 0.0}, {"A": 1.0}, {"A": 10.0}, 89.0)
    assert rm.firm_state == "KILLED" and rm.firm_scale() == 0.0
    rm.post_mark(3 + rm.limits.firm_kill_days, "d", {"A": 0.0}, {"A": 1.0}, {"A": 10.0}, 89.0)
    assert rm.firm_state == "RECOVERY" and rm.firm_scale() == rm.limits.firm_recovery_scale


def test_desk_daily_loss_limit_halts_desk():
    rm = RiskManager(["A"], RiskLimits())
    halted, _ = rm.post_mark(5, "d", {"A": -5.0}, {"A": 1.0}, {"A": 100.0}, 1e6)
    assert halted == ["A"]
    assert rm.desk_scale("A", 6) == 0.0
    assert rm.desk_scale("A", 5 + rm.limits.desk_daily_loss_halt_days) == 1.0


def test_desks_flat_when_budget_zero():
    m = SyntheticMarket(MarketConfig(n_days=200, seed=4))
    for d in (CTADesk(), VolDesk(), MarketMakingDesk()):
        for t in range(150):
            v = m.view(t)
            if t:
                d.mark(v)
            d.observe(v)
        assert not d.target(m.view(149), 0.0).any()


def test_end_to_end_run(small_run):
    m, r = small_run
    assert len(r.equity) == m.n_days
    assert np.isfinite(r.equity).all()
    assert np.isclose(r.equity[-1] - r.equity[0], r.desk_pnl.sum())
    assert (r.desk_cost >= 0).all()
    s = summarize(r)
    assert set(s["desks"]) == set(r.desk_names)
    md = to_markdown(r, s)
    assert "페이퍼" in md


def test_realized_risk_near_budget(small_run):
    _, r = small_run
    f = summarize(r)["firm"]
    # 목표 10% 대비 과도한 리스크를 쓰지 않는다
    assert f["ann_vol"] < 0.15


def test_vol_desk_stress_limit_caps_crash_loss():
    m = SyntheticMarket(MarketConfig(n_days=60, seed=5))
    v = m.view(59)
    d = VolDesk()
    rm = RiskManager(["VOL"], RiskLimits())
    big = np.full(3, 1e6)  # 과도한 분산 매도
    capped = rm.stress_check(59, "d", d, big, v, 100e6)
    assert d.stress_loss(capped, v) <= rm.limits.desk_stress_limit * 100e6 + 1e-6
    assert rm.events[-1].kind == "STRESS_LIMIT"
    assert d.stress_loss(-big, v) == 0.0  # 분산 매수(롱 볼)는 급락 시 이익
