# GPT-OSS Round 2 Plan

## 목표

- `seed_v2_all` 기준 1차 LoRA에서 확인된 도메인 이탈과 일반론 응답을 줄입니다.
- 현재 운영 기준은 `adapter + vLLM + OpenWebUI` 검증이며, 2차도 같은 경로로 먼저 확인합니다.

## 1차 실패 요약

- `validation_result.json` 기준으로 일부 질문은 자연스러운 한국어 답을 생성했지만, 핵심 도메인 질문에서 다른 도메인 설명으로 이탈했습니다.
- 특히 `유형분류 기준 순서`, `평가 결과 해석 주의점` 같은 질문에서 제로인 방법론이 아니라 일반 NLP/평가 지표 설명으로 붕 뜨는 현상이 보였습니다.
- 학습 리포트의 샘플 추론에도 `재무제표`, `데이터 분석`, `정확도/재현율`처럼 원문과 무관한 표현이 섞여 있습니다.

## 원인 가설

- `60 step`은 `seed_v2_all` 전체 기준으로 너무 짧아 도메인 정착 전 단계에서 끝났을 가능성이 높습니다.
- `q_proj`, `v_proj`만으로는 현재 데이터셋의 폭을 흡수하기에 표현력이 부족할 수 있습니다.
- `max_length 512`는 `coverage` 샘플과 표/규칙 설명이 긴 섹션에서 문맥 손실을 만들 수 있습니다.
- 현재 검증 질문셋은 도메인 핵심 규칙 일부를 잘 찌르지만, `유형 BM`, `환헷지`, `등급 산출 지표` 같은 기준형 질문은 부족합니다.

## 2차 학습 설정

- config: `configs/gpt_oss_20b_seed_v2_all_round2.json`
- 변경 방향:
  - `max_length`: `512 -> 1024`
  - `gradient_accumulation_steps`: `1 -> 4`
  - `max_steps`: `60 -> 240`
  - `learning_rate`: `2e-4 -> 1e-4`
  - `lora_r`: `8 -> 16`
  - `lora_alpha`: `16 -> 32`
  - `target_modules`: `q_proj`, `v_proj` -> `q_proj`, `k_proj`, `v_proj`, `o_proj`

## 검증 질문셋

- `펀드 평가에서 유형분류가 왜 중요한가요?`
- `국내투자 펀드와 국외투자 펀드는 유형분류 기준을 어떻게 다르게 적용하나요?`
- `유형생성의 기본원칙인 충분성, 비교성, 지속성을 설명해줘.`
- `펀드 %순위와 등급의 평가대상은 어떤 기준으로 선정하나요?`
- `제로인 펀드등급은 어떤 지표를 바탕으로 산출하나요?`
- `유형별 벤치마크는 왜 필요하고 국내펀드와 해외펀드에는 어떻게 적용하나요?`
- `해외펀드의 환헷지 여부는 BM 적용에서 어떻게 구분하나요?`

## 실행 순서

1. `python scripts/run_smoke_gpt_oss_20b.py --config configs/gpt_oss_20b_seed_v2_all_round2.json`
2. `python scripts/check_gpt_oss_model_output.py --config configs/gpt_oss_20b_seed_v2_all_round2.json --model-path unsloth/gpt-oss-20b-BF16 --adapter-path llm_model_lora/gpt-oss-20b-seed-v2-all-round2 --report-path llm_model_lora/gpt-oss-20b-seed-v2-all-round2/adapter_validation.json`
3. `ADAPTER_NAME=gpt-oss-20b-seed-v2-all-round2 ADAPTER_PATH=llm_model_lora/gpt-oss-20b-seed-v2-all-round2 bash scripts/run_vllm_adapter_server.sh`
4. `python scripts/check_vllm_adapter_server.py --port 8000 --model gpt-oss-20b-seed-v2-all-round2`
5. `OpenWebUI`에서 대표 질문 7개를 재질의해 비교합니다.

## 통과 기준

- 7개 질문 중 최소 5개에서 제로인 방법론 범위 안의 답을 생성
- `정확도`, `재현율`, `데이터 분석`, `재무제표` 같은 타 도메인 표현이 사라질 것
- `유형분류`, `평가대상`, `등급`, `BM`, `환헷지`, `CE/ZI` 중 질문 대상 개념을 답변에 직접 반영할 것
- `OpenWebUI`와 로컬 API 호출 결과가 같은 모델 ID로 안정적으로 동작할 것

## 2차 실행 결과

- 학습 완료:
  - `output_dir`: `llm_model_lora/gpt-oss-20b-seed-v2-all-round2`
  - `train_runtime`: 약 `841s`
  - `train_loss`: 약 `1.009`
- 직접 검증 결과:
  - 1차에서 보였던 `정확도/재현율`, `재무제표`, `데이터 분석` 같은 완전 다른 도메인 이탈은 일부 줄었습니다.
  - 하지만 답변이 여전히 제로인 원문 규칙에 정확히 anchored 되지 않고, 일반 금융/투자 상식 수준으로 평탄화되는 문제가 남아 있습니다.
  - 일부 샘플 추론에서는 `포맷`, `제약`, `주제` 같은 메타 문구를 그대로 답하는 현상도 확인됐습니다.

## 현재 판단

- `round2`는 1차보다 한국어 안정성과 일반적인 관련성은 좋아졌습니다.
- 그러나 `유형분류 기준`, `유형생성의 기본원칙`, `평가대상 선정 기준`, `해외펀드 표`, `BM 적용`처럼 문서 고유 규칙을 정확히 재현하는 수준에는 아직 못 미칩니다.
- 따라서 현재 시점에서는 `round2 adapter`를 기본 서비스 모델로 승격하지 않고, 기존 `gpt-oss-20b-seed-v2-all` 운영선을 유지하는 편이 안전합니다.

## 다음 수정 우선순위

1. `seed_v2_all` canonical 데이터에서 `주제`, `핵심 주제`, `추가 요소`, `포맷`, `제약` 같은 메타성 표현을 줄입니다.
2. `01_유형분류_기준`, `02_펀드평가_방법론`의 대표 규칙형 샘플을 더 직접적인 질문/답변으로 재작성합니다.
3. 표 설명 샘플에서 실제 열/행/기준값을 더 정확히 풀고, 일반론 문장을 줄입니다.
4. 그 뒤 `round3`는 데이터 정비 후 다시 짧게 검증하는 편이 좋습니다.
