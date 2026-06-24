# V8 Worktree Prep Design

**Goal:** `feature-gen-dataset-v7`를 기반으로 새 워크트리 `feature-gen-dataset-v8`를 만들되, 학습 코드와 운영 흐름은 유지하고 기존 라운드 산출물과 로그는 비워서 깨끗한 시작점을 만든다.

## Scope

- 새 git worktree `feature-gen-dataset-v8` 생성
- `v7`의 스크립트, 테스트, 설정 패턴, 에이전트 정의 계승
- 기존 `round2`~`round5` 학습 산출물과 smoke 로그 제거
- `v8` 출발점 문서 정리

## Keep

- `.cursor/agents/`
- `scripts/`
- `tests/`의 실행 스크립트와 테스트 코드
- `docs/`
- `configs/`
- `llm_datasets/seed_v7/`의 참고가 되는 데이터 생성 패턴

## Reset Or Remove

- `llm_model_lora/`의 round 산출물
- `tests/log/`의 기존 smoke 리포트
- `v7` 명칭에 직접 묶인 문서 서술 중 `v8` 출발에 혼선을 주는 내용

## Success Criteria

- 새 워크트리가 `.worktrees/feature-gen-dataset-v8`에 생성된다.
- `v8` 작업 디렉터리에 이전 round 어댑터와 로그가 남아 있지 않다.
- `v8`에서 다음 데이터 소스와 `round1` 준비를 바로 시작할 수 있다.
