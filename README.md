## dev_data/fine-tuning (프로젝트 작업 루트)

이 디렉토리는 제로인 방법론 기반 LLM 파인튜닝 프로젝트의 **작업 루트(working directory)** 입니다.  
본 프로젝트에서 생성되는 **모든 결과물/중간 산출물은 반드시 이 하위에만 저장**합니다.

### 폴더 구조

- `datasets/`
  - `canonical/`: 원천 데이터(일반적인 chat `messages` JSONL)
  - `splits/`: train/val/test + hallucination(범위밖/거절) 전용 분할본
  - `rendered/`: `openai-harmony`로 렌더링한 학습 입력
  - `reports/`: 데이터셋 통계/검증 리포트
- `models/`
  - `adapters/`: LoRA 어댑터
  - `merged/`: (선택) merge 모델
  - `quantized/`: (선택) 양자화 모델
  - `runs/`: 학습 실행 설정/로그 스냅샷
- `evals/`
  - `sets/`: 평가 질문/정답 템플릿
  - `results/`: 평가 결과(자동/휴먼)
- `serving/`
  - `vllm/`: vLLM 서빙 설정/런북
  - `configs/`: 프로덕션/스테이징 파라미터

### 원문 문서 위치

학습 원문 문서는 `dev_data/docs/제로인방법론/` 아래에 존재하며, 결과물은 이 폴더로 복사하지 않고 **참조만**합니다.

