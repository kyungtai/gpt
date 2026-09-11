# Market Making

Codex와 Claude Code가 GitHub를 통해 함께 사용하는 마켓메이킹 프로젝트입니다.

## 협업 방식

- 공통 작업 규칙은 `AGENTS.md`에 둡니다. Claude Code는 `CLAUDE.md`를 통해 같은 규칙을 읽습니다.
- 동시에 작업할 때는 각 도구가 별도의 clone 또는 Git worktree를 사용합니다.
- 작업 브랜치는 `codex/<작업명>`, `claude/<작업명>`으로 구분하고 Pull Request로 합칩니다.
- 작업 시작 전 최신 변경을 fetch하고, 커밋에는 작업 내용과 검증 결과를 남깁니다.

현재는 협업용 초기 구조만 있으며 거래 전략이나 실행 코드는 없습니다.
