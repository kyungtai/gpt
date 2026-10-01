"""운영 리포트: 성과 지표, 데스크 기여, 리스크 이벤트, 월별 수익률."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .firm import Results
from .market import TRADING_DAYS


def _stats(pnl: np.ndarray, capital: float) -> dict:
    sd = pnl.std()
    cum = np.cumsum(pnl)
    dd = np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum) if len(cum) else 0.0
    return {
        "pnl": pnl.sum(),
        "ann_ret": pnl.mean() * TRADING_DAYS / capital,
        "ann_vol": sd * np.sqrt(TRADING_DAYS) / capital,
        "sharpe": pnl.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0,
        "max_dd": dd / capital,
        "hit": float((pnl > 0).sum() / max((pnl != 0).sum(), 1)),
        "worst": pnl.min(),
        "best": pnl.max(),
    }


def summarize(r: Results) -> dict:
    eq = r.equity
    daily_ret = np.diff(eq) / eq[:-1]
    years = (len(eq) - 1) / TRADING_DAYS
    peak = np.maximum.accumulate(eq)
    firm = {
        "start": str(r.dates[0]),
        "end": str(r.dates[-1]),
        "days": len(eq) - 1,
        "capital": r.capital,
        "final_equity": eq[-1],
        "total_return": eq[-1] / eq[0] - 1,
        "cagr": (eq[-1] / eq[0]) ** (1 / years) - 1 if years > 0 else 0.0,
        "ann_vol": daily_ret.std() * np.sqrt(TRADING_DAYS),
        "sharpe": daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS),
        "max_dd": float(np.max(1 - eq / peak)),
        "worst_day": daily_ret.min(),
        "best_day": daily_ret.max(),
        "hit": float((daily_ret > 0).mean()),
        "costs": r.desk_cost.sum(),
        "orders": r.n_orders,
        "mm_fills": r.mm_fills,
        "days_by_state": dict(Counter(r.firm_state[1:])),
    }
    desks = {}
    total_pnl = r.desk_pnl.sum()
    for i, name in enumerate(r.desk_names):
        s = _stats(r.desk_pnl[1:, i], r.capital)
        s["cost"] = r.desk_cost[:, i].sum()
        s["share"] = s["pnl"] / total_pnl if total_pnl else 0.0
        s["avg_budget"] = r.annual_budget[1:, i].mean()
        s["avg_gross_x"] = (r.gross[1:, i] / eq[1:]).mean()
        desks[name] = s
    pnl = r.desk_pnl[1:]
    live = pnl.std(axis=0) > 0
    corr = np.full((len(r.desk_names),) * 2, np.nan)
    if live.sum() > 1:
        corr[np.ix_(live, live)] = np.corrcoef(pnl[:, live].T)
    events = Counter((e.scope, e.kind) for e in r.events)
    return {"firm": firm, "desks": desks, "corr": corr, "events": events}


def monthly_returns(r: Results) -> list[tuple[str, float]]:
    months = r.dates.astype("datetime64[M]")
    out = []
    for mth in np.unique(months):
        idx = np.flatnonzero(months == mth)
        start = r.equity[max(idx[0] - 1, 0)]
        out.append((str(mth), r.equity[idx[-1]] / start - 1))
    return out


def to_markdown(r: Results, s: dict | None = None) -> str:
    s = s or summarize(r)
    f = s["firm"]
    L = []
    L.append(f"# {r.firm} — 운영 리포트")
    L.append("")
    L.append(f"> 합성 시장 데이터 + 페이퍼 브로커 결과입니다. 실제 시장 성과가 아니며 실주문은 발생하지 않았습니다.")
    L.append("")
    L.append(f"기간: {f['start']} ~ {f['end']} ({f['days']} 영업일)")
    L.append("")
    L.append("## 펌 요약")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---:|")
    rows = [
        ("초기 자본", f"${f['capital'] / 1e6:,.1f}M"),
        ("최종 자본", f"${f['final_equity'] / 1e6:,.1f}M"),
        ("누적 수익률", f"{f['total_return']:.1%}"),
        ("CAGR", f"{f['cagr']:.1%}"),
        ("연율 변동성", f"{f['ann_vol']:.1%}"),
        ("샤프", f"{f['sharpe']:.2f}"),
        ("최대 낙폭", f"{f['max_dd']:.1%}"),
        ("최악/최고 일간", f"{f['worst_day']:.2%} / {f['best_day']:.2%}"),
        ("일간 승률", f"{f['hit']:.0%}"),
        ("총 거래비용", f"${f['costs'] / 1e6:,.2f}M"),
        ("리밸런스 주문 수 / MM 체결 수", f"{f['orders']:,} / {f['mm_fills']:,}"),
        ("펌 상태 일수", ", ".join(f"{k} {v}" for k, v in f["days_by_state"].items())),
    ]
    L += [f"| {a} | {b} |" for a, b in rows]
    L.append("")
    L.append("## 데스크별 성과 (자본 대비)")
    L.append("")
    L.append("| 데스크 | 손익 | 기여 | 연수익 | 연변동성 | 샤프 | MDD | 승률 | 비용 | 평균예산(연) | 평균그로스 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d in s["desks"].items():
        L.append(
            f"| {name} | ${d['pnl'] / 1e6:,.2f}M | {d['share']:.0%} | {d['ann_ret']:.2%} | {d['ann_vol']:.2%} | "
            f"{d['sharpe']:.2f} | {d['max_dd']:.2%} | {d['hit']:.0%} | ${d['cost'] / 1e6:,.2f}M | "
            f"${d['avg_budget'] / 1e6:,.2f}M | {d['avg_gross_x']:.1f}x |"
        )
    L.append("")
    L.append("## 데스크 손익 상관")
    L.append("")
    names = r.desk_names
    L.append("| | " + " | ".join(names) + " |")
    L.append("|---|" + "---:|" * len(names))
    for i, n in enumerate(names):
        L.append(f"| {n} | " + " | ".join("" if np.isnan(c) else f"{c:.2f}" for c in s["corr"][i]) + " |")
    L.append("")
    L.append("## 리스크 이벤트")
    L.append("")
    if s["events"]:
        L.append("| 범위 | 종류 | 횟수 |")
        L.append("|---|---|---:|")
        for (scope, kind), c in sorted(s["events"].items()):
            L.append(f"| {scope} | {kind} | {c} |")
        L.append("")
        major = [e for e in r.events if e.kind not in ("GROSS_CAP", "VAR_SCALE", "STRESS_LIMIT")]
        if major:
            L.append("주요 이벤트 (그로스/VaR/스트레스 스케일 제외):")
            L.append("")
            for e in major[:40]:
                L.append(f"- {e.date} `{e.scope}` **{e.kind}** — {e.detail}")
            if len(major) > 40:
                L.append(f"- … 외 {len(major) - 40}건")
    else:
        L.append("없음")
    L.append("")
    if r.allocations:
        a = r.allocations[-1]
        L.append(f"## 최근 CIO 배분 ({a['date']})")
        L.append("")
        L.append("| 데스크 | 비중 | 126일 샤프 | 연간 리스크 예산 |")
        L.append("|---|---:|---:|---:|")
        for i, n in enumerate(names):
            sr = a["sharpe"][i]
            L.append(f"| {n} | {a['weights'][i]:.0%} | {'' if np.isnan(sr) else f'{sr:.2f}'} | ${a['budgets'][i] / 1e6:,.2f}M |")
        L.append("")
    L.append("## 월별 수익률")
    L.append("")
    L.append("| 월 | 수익률 |")
    L.append("|---|---:|")
    for mth, ret in monthly_returns(r):
        L.append(f"| {mth} | {ret:+.2%} |")
    L.append("")
    return "\n".join(L)


def write_outputs(r: Results, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(r)
    md = out / "report.md"
    md.write_text(to_markdown(r, s), encoding="utf-8")
    daily = out / "daily_pnl.csv"
    with daily.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "equity", "state", *[f"pnl_{n}" for n in r.desk_names], *[f"budget_{n}" for n in r.desk_names]])
        for t in range(len(r.equity)):
            w.writerow(
                [str(r.dates[t]), f"{r.equity[t]:.2f}", r.firm_state[t]]
                + [f"{x:.2f}" for x in r.desk_pnl[t]]
                + [f"{x:.2f}" for x in r.annual_budget[t]]
            )
    ev = out / "risk_events.csv"
    with ev.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "scope", "kind", "detail"])
        for e in r.events:
            w.writerow([e.date, e.scope, e.kind, e.detail])
    return {"report": md, "daily": daily, "events": ev}
