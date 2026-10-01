"""CLI: python -m propfirm run [--days N] [--seed S] [--capital C] [--out DIR]"""

from __future__ import annotations

import argparse

from .firm import Firm, FirmConfig
from .market import MarketConfig, SyntheticMarket
from .report import summarize, write_outputs


def main(argv=None):
    p = argparse.ArgumentParser(prog="propfirm", description="멀티전략 프랍 펌 시뮬레이션 (페이퍼 전용)")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="합성 시장에서 펌을 운영하고 리포트를 생성")
    run.add_argument("--days", type=int, default=756)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--capital", type=float, default=100e6)
    run.add_argument("--target-vol", type=float, default=0.10)
    run.add_argument("--out", default="out/latest")
    run.add_argument("--quiet", action="store_true")
    sc = sub.add_parser("scenarios", help="여러 시드(시장 시나리오)에서 펌 성과 분포 점검")
    sc.add_argument("--n", type=int, default=20)
    sc.add_argument("--days", type=int, default=756)
    args = p.parse_args(argv)
    if args.cmd == "scenarios":
        return scenarios(args.n, args.days)

    market = SyntheticMarket(MarketConfig(n_days=args.days, seed=args.seed))
    firm = Firm(market, FirmConfig(capital=args.capital, target_vol=args.target_vol))
    res = firm.run(progress=not args.quiet)
    paths = write_outputs(res, args.out)
    f = summarize(res)["firm"]
    print(
        f"\n최종 자본 ${f['final_equity'] / 1e6:,.1f}M | 누적 {f['total_return']:.1%} | 샤프 {f['sharpe']:.2f} | "
        f"변동성 {f['ann_vol']:.1%} | MDD {f['max_dd']:.1%}"
    )
    for k, v in paths.items():
        print(f"  {k}: {v}")


def scenarios(n: int, days: int):
    import numpy as np

    rows = []
    for seed in range(1, n + 1):
        res = Firm(SyntheticMarket(MarketConfig(n_days=days, seed=seed))).run()
        s = summarize(res)
        f = s["firm"]
        rows.append((seed, f["cagr"], f["sharpe"], f["ann_vol"], f["max_dd"], f["days_by_state"].get("KILLED", 0),
                     {k: d["sharpe"] for k, d in s["desks"].items()}))
        print(f"seed {seed:3d} | CAGR {f['cagr']:6.1%} | 샤프 {f['sharpe']:5.2f} | 변동성 {f['ann_vol']:5.1%} | "
              f"MDD {f['max_dd']:5.1%} | 킬스위치일 {rows[-1][5]}")
    arr = np.array([r[1:5] for r in rows])
    print("\n분포 (중앙값 / 5% / 95%)")
    for j, name in enumerate(["CAGR", "샤프", "변동성", "MDD"]):
        q = np.percentile(arr[:, j], [50, 5, 95])
        print(f"  {name:4s} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}")
    print("데스크 샤프 중앙값 / 음수 비율:")
    for k in rows[0][6]:
        x = np.array([r[6][k] for r in rows])
        print(f"  {k:9s} {np.median(x):5.2f}  {np.mean(x < 0):4.0%}")


if __name__ == "__main__":
    main()
