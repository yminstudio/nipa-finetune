# Zeroin Training Validation Reference

## Typical Commands

### 1. 데이터 파이프라인

`v5` 오케스트레이터가 있으면 우선 사용합니다.

```bash
python scripts/gen_dataset_v5/10_run_v5_all_chapters.py \
  --phase data \
  --round-label <round-label> \
  --force
```

### 2. Harmony 렌더링

```bash
python llm_datasets/render_gpt-oss-harmony.py \
  --input <seed-jsonl>
```

### 3. LoRA 학습

```bash
python scripts/run_smoke_gpt_oss_20b.py \
  --config <training-config.json>
```

### 4. 직접 adapter 검증

```bash
python scripts/check_gpt_oss_model_output.py \
  --config <training-config.json> \
  --adapter-path <adapter-output-dir>
```

### 5. vLLM adapter 서빙

```bash
PORT=<port> \
ADAPTER_NAME=<served-model-name> \
ADAPTER_PATH="<adapter-output-dir>" \
bash scripts/run_vllm_adapter_server.sh
```

### 6. vLLM 헬스체크

```bash
python scripts/check_vllm_adapter_server.py \
  --port <port> \
  --model <served-model-name>
```

## Round+1 Decision Rules

### round 이름

- 사용자가 명시한 round 이름이 있으면 그대로 사용합니다.
- 없으면 최신 round 다음 번호를 사용합니다.
- 새 round의 `dataset_path`, `prompt_source_path`, `output_dir`, `validation_report_path`, `served_model_name`은 모두 같은 round를 가리켜야 합니다.

### 이어학습 여부

- 사용자가 `round+1` 학습을 요청하고 이전 adapter 품질이 치명적으로 나쁘지 않았다면 이어학습을 우선 검토합니다.
- 이전 adapter 품질이 명확히 부적합하거나 사용자가 새로 시작하라고 요청한 경우에만 base에서 다시 시작합니다.

### merge 여부

- 기본값은 `merge 하지 않음`입니다.
- 사용자가 명시적으로 요청할 때만 merge 단계를 추가합니다.

## p6 Review Checklist

다음 질문으로 샘플 출력과 validation 결과를 검토합니다.

### 공통

- 답이 문서 기반인가
- 답이 독립적으로 이해 가능한가
- 자연스럽게 연결되지만 새로운 정보가 생기지 않았는가
- 질문에 없는 대상, 비교, 예외가 추가되지 않았는가
- 일반 금융 지식이나 관행으로 빈칸을 메우지 않았는가

### definition

- 정의문이 실제로 포함되는가
- 절차/예외/비교가 질문 없이 덧붙지 않는가

### criteria

- 판단 기준만 남는가
- 적용 절차나 비교 설명으로 새지 않는가

### comparison

- 필요한 비교축만 다루는가
- 각 대상의 정의/절차 설명이 불필요하게 붙지 않는가

### application

- 적용 순서, 조건, 전환 규칙 중심인가
- 정의 재설명이나 일반 배경설명이 과하지 않은가

## Result Summary Template

```text
Round:
- 대상 round:
- 데이터 범위:

Training:
- canonical JSONL:
- harmony:
- config:
- adapter output:

Validation:
- 직접 검증:
- p6 평가:
- vLLM 검토:

Risks:
- 남은 리스크 1
- 남은 리스크 2
```
