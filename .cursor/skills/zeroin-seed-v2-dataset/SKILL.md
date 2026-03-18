---
name: zeroin-seed-v2-dataset
description: Builds `llm_datasets/seed_v2` training datasets from Markdown files in `docs/제로인방법론/divide`, validates them, renders GPT-OSS Harmony data, and prepares follow-up GPT-OSS training and validation steps. Use when creating, expanding, validating, revising, or continuing the 제로인 방법론 seed_v2 training pipeline.
---

# Zeroin Seed V2 Dataset

## Goal

`docs/제로인방법론/divide`의 문서를 `llm_datasets/seed_v2` 아래의 학습용 JSONL 데이터로 변환합니다.

기본 원칙:

- 출력은 `seed_v1`과 호환되는 `messages` 중심 JSONL 형식을 사용합니다.
- 기본 샘플 단위는 `섹션/소제목 단위`입니다.
- 문서가 짧거나 문서 전체 맥락이 중요하면 `문서 전체 요약/개요 샘플`을 추가합니다.
- 상위 제목 샘플은 자신의 하위 제목 내용을 모두 포괄해야 합니다.
- 답변은 항상 한국어로 작성합니다.
- 원문 문장을 길게 베끼지 말고, 방법론 규칙을 학습 가능한 Q/A 형태로 재구성합니다.
- 다른 원칙보다 `상위 제목이 하위 제목 전체를 포함해 요약되는가`를 우선합니다.

## Quick Start

1. 문서 목록과 출력 대상 manifest를 만듭니다.
2. manifest를 기준으로 문서별 샘플을 작성합니다.
3. `seed_v1` 호환 JSONL로 저장합니다.
4. 검증 스크립트로 형식을 검사합니다.
5. 필요하면 `llm_datasets/render_gpt-oss-harmony.py`로 Harmony 렌더링을 수행합니다.
6. 사용자가 계획 진행을 요청하면 `configs/gpt_oss_20b_seed_v2_all.json` 기준으로 학습/병합/직접 검증 단계까지 연결합니다.

## Commands

문서 manifest 생성:

```bash
python .cursor/skills/zeroin-seed-v2-dataset/scripts/build_manifest.py \
  --docs-dir docs/제로인방법론/divide \
  --output llm_datasets/seed_v2/doc_manifest.jsonl
```

JSONL 검증:

```bash
python .cursor/skills/zeroin-seed-v2-dataset/scripts/validate_seed_v2.py \
  --input llm_datasets/seed_v2/seed_divide_all.jsonl
```

외부 OpenAI API로 재작성:

```bash
python .cursor/skills/zeroin-seed-v2-dataset/scripts/rewrite_seed_v2_with_openai.py \
  --input llm_datasets/seed_v2/seed_v2_02_펀드평가_방법론.jsonl \
  --doc docs/제로인방법론/divide/02_펀드평가_방법론.md \
  --output llm_datasets/seed_v2/seed_v2_02_펀드평가_방법론.jsonl
```

기본적으로 저장소 루트의 `.env`에서 `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_API_BASE`를 읽습니다.

Harmony 렌더링:

```bash
python llm_datasets/render_gpt-oss-harmony.py \
  --input llm_datasets/seed_v2/seed_divide_all.jsonl
```

LoRA 학습:

```bash
python scripts/run_smoke_gpt_oss_20b.py \
  --config configs/gpt_oss_20b_seed_v2_all.json
```

LoRA 병합:

```bash
python scripts/merge_gpt_oss_lora.py \
  --config configs/gpt_oss_20b_seed_v2_all.json
```

직접 검증:

```bash
python scripts/check_gpt_oss_model_output.py \
  --config configs/gpt_oss_20b_seed_v2_all.json
```

## Workflow

다음 체크리스트를 그대로 따릅니다.

```text
Seed v2 Progress
- [ ] 1. 입력 문서 inventory 생성
- [ ] 2. 문서별 샘플 범위 결정
- [ ] 3. 상위 제목 coverage Q/A 작성
- [ ] 4. 섹션 단위 Q/A 작성
- [ ] 5. 표/목록/임계값 설명 Q/A 보강
- [ ] 6. 문서 레벨 요약 샘플 보강
- [ ] 7. JSONL 저장
- [ ] 8. 스키마 검증
- [ ] 9. Harmony 렌더링
```

### 1. 입력 문서 inventory 생성

- `docs/제로인방법론/divide/*.md` 전체를 스캔합니다.
- `목차.md`, `개정사항.md`도 포함할 수 있지만, 과대표집하지 않습니다.
- 기본 출력 위치는 `llm_datasets/seed_v2/`입니다.

### 2. 문서별 샘플 범위 결정

문서마다 아래 우선순위로 샘플을 만듭니다.

- `coverage`: 상위 제목이 하위 제목 전체를 설명하는 종합형
- `definition`: 용어 정의, 유형 정의, 지표 정의
- `rule`: 분류 규칙, 임계값, 우선순위, 적용 조건
- `procedure`: 단계별 계산 절차, 적용 순서
- `formula`: 수식 의미, 변수 정의, 계산법
- `comparison`: 유형 간 차이, 방법론 간 차이
- `case`: 조건을 주고 유형/판단 결과를 묻는 사례형
- `table_explainer`: 표의 열/행/구분 기준/예외를 설명하는 표 해설형
- `concept`: 왜 필요한가, 어떤 목적을 가지는가

기본 목표:

- 각 `##` 제목마다 하위 `###`, `####`를 모두 포괄하는 `coverage` 샘플 최소 1개
- 각 `###` 제목마다 하위 `####`가 있으면 이를 포괄하는 `coverage` 샘플 최소 1개
- 각 의미 있는 섹션마다 최소 1개 샘플
- 규칙/수식이 많은 섹션은 2~4개 샘플
- 표가 있는 섹션은 표 설명용 샘플 최소 1개 추가
- 문서 전체 설명이 필요한 경우 문서 레벨 개요 샘플 1개 추가

### 3. 제목 계층 coverage 규칙

heading은 자기 자신만 설명하면 부족합니다. 항상 하위 제목까지 포함해 묻고 답할 수 있어야 합니다.

- `##` 제목에 대한 Q/A는 그 아래의 모든 `###`, `####` 내용을 종합해 설명할 수 있어야 합니다.
- `###` 제목에 대한 Q/A는 그 아래의 모든 `####` 내용을 종합해 설명할 수 있어야 합니다.
- 하위 제목별 세부 샘플과 별개로, 상위 제목 coverage 샘플을 반드시 따로 만듭니다.
- 제목이 `@docs/제로인방법론/divide/01_유형분류_기준.md:1`처럼 상위 개념을 나타내면, 답변은 하위 유형, 기준, 예외, 적용 구조까지 함께 요약해야 합니다.

coverage 샘플 질문 예시:

- "`1. 유형분류 기준`의 전체 구조와 핵심 판단 기준을 설명해줘."
- "`2. 펀드평가 방법론` 아래에서 평가대상, 등급, 수익률 평가를 어떻게 연결해서 이해해야 하나요?"

### 4. 표 설명 데이터 생성 규칙

표는 본문에 흡수하지 말고, 필요하면 별도 샘플로 분리해 학습 데이터에 반영합니다.

- `####` 아래 표가 있으면 표를 설명하는 Q/A를 최소 1개 이상 생성합니다.
- `###`나 `##` 아래 직접 표가 있는 경우도 동일하게 적용합니다.
- 표 설명 샘플은 최소한 아래 중 2개 이상을 담아야 합니다.
- 표의 열/행 의미
- 구분 기준 또는 임계값
- 항목 간 차이
- 적용 순서 또는 우선순위
- 예외 조건
- 표가 분류표라면 "어떻게 분류하는가"를 묻는 사례형 샘플을 추가합니다.
- 표가 비교표라면 "항목 간 차이"를 묻는 비교형 샘플을 추가합니다.
- 표가 임계값표라면 숫자 기준을 묻는 규칙형 샘플을 추가합니다.

### 5. 샘플 작성 규칙

- `system` 메시지는 기존 `seed_v1`의 도메인 어시스턴트 톤을 유지합니다.
- `user`는 실제 질의처럼 자연스럽게 씁니다.
- `assistant`는 문서의 규칙을 요약하고 재구성해 답합니다.
- 긴 수식은 필요 시 의미 중심으로 풀고, 변수 정의 샘플을 별도로 분리합니다.
- coverage 샘플은 하위 항목을 빠뜨리지 않도록 개요, 하위 분류, 핵심 기준, 예외를 함께 담습니다.
- 표 설명 샘플은 표를 보지 않아도 이해되도록 행/열 의미를 문장으로 풀어 씁니다.

### 6. 문서 레벨 요약 샘플

아래 경우에는 섹션 샘플 외에 문서 레벨 샘플을 추가합니다.

- 서문처럼 전체 목적 설명이 중요한 문서
- 한 문서 안에 여러 하위 규칙이 있어 큰 그림 요약이 필요한 문서
- 목차/개정사항처럼 탐색 또는 버전 맥락 자체가 핵심인 문서

### 7. 저장 규칙

- 최종 산출물은 JSONL 한 줄당 1개 레코드로 저장합니다.
- 기본 파일명은 `llm_datasets/seed_v2/seed_divide_all.jsonl`을 우선 사용합니다.
- 필요하면 문서별 보조 파일을 추가로 만들 수 있습니다.

### 8. 검증 규칙

- 모든 레코드는 `id`, `messages`, `meta`를 가져야 합니다.
- `meta.dataset_version`은 `v2`여야 합니다.
- `meta.source.doc_path`는 반드시 `docs/제로인방법론/divide/` 하위 경로여야 합니다.
- 검증 스크립트 통과 전에는 렌더링이나 학습 단계로 진행하지 않습니다.

검수 체크:

- 각 `##` 제목에 coverage 샘플이 있는가
- 각 `###` 제목 아래 `####`가 있으면 coverage 샘플이 있는가
- 표가 있는 heading마다 표 설명 샘플이 있는가
- 표의 기준값, 구분값, 예외가 샘플에 실제로 반영되었는가
- 상위 제목 샘플이 하위 제목 내용을 누락하지 않았는가

## Post-Validation Pipeline

canonical JSONL 품질이 충분히 확인되면 아래 기본 흐름을 따릅니다.

1. `llm_datasets/render_gpt-oss-harmony.py`로 Harmony 렌더링
2. `scripts/run_smoke_gpt_oss_20b.py --config configs/gpt_oss_20b_seed_v2_all.json`
3. 필요하면 adapter 상태를 `scripts/check_gpt_oss_model_output.py --adapter-path ...`로 직접 비교
4. `scripts/merge_gpt_oss_lora.py --config configs/gpt_oss_20b_seed_v2_all.json`
5. `scripts/check_gpt_oss_model_output.py --config configs/gpt_oss_20b_seed_v2_all.json`
6. 이후에만 `scripts/run_vllm_model_server.sh`로 병합 BF16을 서빙

기본 원칙:

- 사용자가 "계획 진행", "다음 단계 진행"처럼 넓게 요청하면 `seed_v2_all` 경로를 우선 기준으로 삼습니다.
- 이미 완료된 단계가 있으면 그 다음 단계부터 이어갑니다.
- `MXFP4`는 최종 목표지만, 저장소에 검증된 툴체인이 고정되기 전에는 BF16 병합본 검증을 기본선으로 둡니다.

## Output Format

필수 스키마와 권장 필드는 [reference.md](reference.md)를 따릅니다.

## Notes

- `seed_v1` 형식과 최대한 맞추되, `qa_type`, `tags`, `answer_key_points`는 더 풍부하게 써도 됩니다.
- 보조 문서(`목차.md`, `개정사항.md`)는 전체 데이터셋에서 비중이 과도해지지 않게 제한합니다.
- 최종 목적이 `gpt-oss` 학습이라도 canonical 데이터는 항상 `messages`로 유지합니다.
