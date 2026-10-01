# Claude Capital — AI 운영 멀티전략 프랍 트레이딩 펌 (Paper)

Codex와 Claude Code가 GitHub로 함께 운영하는 멀티전략 프랍 트레이딩 펌입니다.
**현재는 합성 시장 데이터 + 페이퍼 브로커 전용**이며, 실주문 코드는 없습니다 (`docs/FIRM_CHARTER.md` 5장 참고).

## 구조

| 경로 | 역할 |
|---|---|
| `propfirm/market.py` | 합성 멀티에셋 시장 (선물 13종, 주식 30종목, 국채 커브, 지수 IV, 베이시스, 장중 경로) |
| `propfirm/desks.py` | 6개 데스크: DELTA1, MM, STATARB, VOL, RATES_RV, CTA |
| `propfirm/allocator.py` | CIO 배분기: 성과 가중 + 상관을 반영한 리스크 버짓 |
| `propfirm/risk.py` | CRO 엔진: 손실·드로다운·스트레스·VaR·그로스 한도, 펌 디리스크/킬스위치 |
| `propfirm/broker.py` | 페이퍼 브로커 (실거래 모드는 예외 발생) |
| `propfirm/firm.py` | 일간 운영 루프 |
| `propfirm/report.py` | 운영 리포트 (Markdown/CSV) |
| `docs/FIRM_CHARTER.md` | 조직, 맨데이트, 리스크 한도, 실거래 전환 게이트 |
| `docs/operations/` | CEO 운영 메모와 리포트 |

## 실행

```bash
pip install -r requirements.txt
python -m propfirm run --seed 7 --out out/latest   # 3년 운영 → out/latest/report.md
python -m propfirm scenarios --n 20                # 시나리오 20개 분포 점검
python -m pytest -q
```

## 협업 방식

- 공통 작업 규칙은 `AGENTS.md`에 둡니다. Claude Code는 `CLAUDE.md`를 통해 같은 규칙을 읽습니다.
- 동시에 작업할 때는 각 도구가 별도의 clone 또는 Git worktree를 사용합니다.
- 작업 브랜치는 `codex/<작업명>`, `claude/<작업명>`으로 구분하고 Pull Request로 합칩니다.
- 작업 시작 전 최신 변경을 fetch하고, 커밋에는 작업 내용과 검증 결과를 남깁니다.
