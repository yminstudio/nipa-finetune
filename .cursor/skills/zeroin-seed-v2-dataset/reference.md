# Seed V2 Reference

## Recommended Output Files

- `llm_datasets/seed_v2/doc_manifest.jsonl`: 입력 문서 inventory
- `llm_datasets/seed_v2/seed_divide_all.jsonl`: 최종 canonical dataset
- `llm_datasets/rendered/gpt-oss/seed_divide_all_harmony.jsonl`: Harmony 렌더링 결과
- `llm_model_lora/gpt-oss-20b-*/train_result.json`: LoRA 학습 리포트
- `llm_model_merged/gpt-oss-20b-*/merge_result.json`: LoRA 병합 리포트
- `llm_model_merged/gpt-oss-20b-*/validation_result.json`: 병합본 직접 검증 리포트

## Required Record Shape

각 JSONL 라인은 아래 구조를 따릅니다.

```json
{
  "id": "zeroin.seed_v2_06_0001",
  "messages": [
    {
      "role": "system",
      "content": "당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트입니다. 한국어로 답변합니다. 상위 제목 질문은 반드시 하위 제목 전체를 포함해 요약합니다."
    },
    {
      "role": "user",
      "content": "추정 주식매도회전율은 어떤 상황에서 계산하나요?"
    },
    {
      "role": "assistant",
      "content": "전월보다 특정 종목의 수량이 감소한 경우를 대상으로, 해당 감소분을 평균 가격과 평균 주식평가액 기준으로 환산해 추정 주식매도회전율을 계산합니다."
    }
  ],
  "meta": {
    "dataset_version": "v2",
    "language": "ko",
    "chapter_id": "06",
    "chapter_title": "포트폴리오분석",
    "section_path": "06>6.2>1",
    "qa_type": "formula",
    "scope": "in_scope",
    "created_at": "2026-03-16T00:00:00Z",
    "source": {
      "doc_path": "docs/제로인방법론/divide/06_포트폴리오분석.md",
      "doc_hash": "fill-me",
      "anchors": [
        "6.2",
        "1) 추정 주식매도회전율"
      ]
    },
    "section_title": "추정 주식매도회전율",
    "difficulty": "medium",
    "answer_key_points": [
      "전월보다 수량이 줄어든 종목 대상",
      "평균 가격과 평균 주식평가액 기준 계산"
    ],
    "tags": [
      "seed",
      "zeroin",
      "chapter06",
      "formula"
    ]
  }
}
```

## Required Meta Fields

- `dataset_version`: 항상 `v2`
- `language`: 항상 `ko`
- `chapter_id`: `00`, `01` ... `08`, `X1`, 또는 보조 문서 식별자
- `chapter_title`: 문서 제목
- `section_path`: 문서 내부 구조를 `>`로 연결
- `qa_type`: 아래 권장 값 중 하나
- `scope`: 기본값 `in_scope`
- `created_at`: ISO-8601 UTC
- `source.doc_path`: 원본 Markdown 경로
- `source.doc_hash`: 가능하면 sha256
- `source.anchors`: 사용한 제목 앵커 목록
- `section_title`: 샘플이 근거한 섹션 제목
- `difficulty`: `easy|medium|hard`
- `answer_key_points`: 답변의 핵심 포인트 배열
- `tags`: 검색/분석용 태그

## Recommended `qa_type`

- `coverage`
- `concept`
- `definition`
- `rule`
- `procedure`
- `formula`
- `comparison`
- `case`
- `table_explainer`

## Document Handling Guidance

- `00_서문.md`: 문서 목적, 범위, 사용 맥락 위주 샘플
- `01`~`08`, `X1`: 섹션 중심 본문 샘플
- `목차.md`: 전체 구조 안내용 샘플 1~2개 정도
- `개정사항.md`: 변경 포인트 요약 샘플 1~2개 정도

## Authoring Rules

- 한 샘플에는 하나의 핵심 질문만 넣습니다.
- 문장 복사보다 규칙 재구성을 우선합니다.
- 숫자 임계값, 우선순위, 예외 조건은 가능한 한 명시합니다.
- 계산식이 복잡하면 "식 자체 설명"과 "변수 의미 설명" 샘플을 나눕니다.
- 사례형(`case`)은 입력 조건, 판단 결과, 근거를 분리해 답변합니다.
- 제목 계층 전체를 포괄하는 요약이 다른 세부 원칙보다 우선합니다.
- `coverage` 샘플은 부모 heading의 하위 heading 전체를 포괄해야 합니다.
- 상위 heading 샘플은 개요만 적지 말고, 하위 분류와 판단 기준까지 포함해야 합니다.
- 표가 있는 heading은 `table_explainer` 샘플을 추가해 표의 행/열 의미, 구분 기준, 예외를 설명합니다.
- 표 기반 샘플은 표를 보지 않아도 이해 가능한 문장형 설명으로 작성합니다.

## Coverage Guidance

- `##` heading 샘플은 그 아래 `###`, `####` 전체를 요약하는 종합 질문/답변으로 작성합니다.
- `###` heading 아래 `####`가 존재하면 별도의 종합형 샘플을 추가합니다.
- 세부 샘플만 있고 상위 종합 샘플이 없으면 coverage 부족으로 간주합니다.

## Table Guidance

- 표가 분류 기준을 담고 있으면 `rule` 또는 `case` 샘플을 함께 만듭니다.
- 표가 비교 목적이면 `comparison` 샘플을 함께 만듭니다.
- 표가 수치 기준을 담고 있으면 숫자 임계값을 명시한 샘플을 만듭니다.
- 하나의 표에서 정보가 많으면 "표 전체 설명"과 "핵심 행/열 설명"을 나눠 2개 이상 샘플로 분리할 수 있습니다.

## Suggested ID Format

```text
zeroin.seed_v2_<chapter>_<serial4>
```

예시:

- `zeroin.seed_v2_01_0001`
- `zeroin.seed_v2_06_0014`
- `zeroin.seed_v2_X1_0003`

## Training Pipeline Defaults

사용자가 학습 계획까지 이어서 진행하길 원하면 기본값은 아래와 같습니다.

- training config: `configs/gpt_oss_20b_seed_v2_all.json`
- training script: `scripts/run_smoke_gpt_oss_20b.py`
- merge script: `scripts/merge_gpt_oss_lora.py`
- direct validation script: `scripts/check_gpt_oss_model_output.py`
- vLLM merged serve script: `scripts/run_vllm_model_server.sh`

우선순위:

1. canonical JSONL 품질 확보
2. Harmony 렌더링
3. LoRA 학습
4. adapter 또는 병합 BF16 직접 검증
5. 병합 BF16 vLLM 서빙
