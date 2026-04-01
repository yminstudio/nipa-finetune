# GPT-OSS Smoke Report

## Summary
- status: `success`
- model_name: `unsloth/gpt-oss-20b-BF16`
- adapter_path: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/llm_model_lora/gpt-oss-20b-seed-v7-single-qa`
- prompt_source_path: `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v7/tests/smoke_v7_single_qa_prompt.jsonl`
- max_new_tokens: `384`

## System Prompt

당신은 제로인 펀드평가 방법론에 근거해 답변하는 도메인 어시스턴트입니다.

## Sample Inference

### single_qa_0001

**Question**

유형생성의 기본원칙에서 충분성, 비교성, 지속성을 정의해

**Answer**

유형 생성의 기본원칙은 개별 대상이 의미 있게 평가되기 위해 필요한 세 가지 조건을 말합니다.

1. 충분성 (Sufficiency)
- 개별 대상에 필요한 정보가 충분히 존재해야 한다는 의미입니다.
- 즉, 대상에 대한 데이터(예: 수익, 위험, 기간 등)가 충분히 수집되어 있어야만 독립된 유형으로 만들 수 있습니다.
- 충분성이 확보되지 않으면 데이터 부족으로 인해 의미 있는 평가가 어렵습니다.

2. 비교성 (Comparability)
- 동일 유형 내의 대상들끼리 서로 비교가 가능해야 한다는 의미입니다.
- 기본성설정된 기준에 따라 여러 대상이 모여 하나의 유형이 되므로, 서로 비교가 가능해야만 그 유형 내에서 일반적인 특성을 파악할 수 있습니다.
- 비교성이 떨어지면 한 유형 내에서 대상들 간에 의미 있는 차이가 없거나 너무 크게 있어 비교가 어렵습니다.

3. 지속성 (Continuity)
- 유형이 일시적인 현상이 아니라 장기적으로 유지될 수 있어야 한다는 의미입니다.
- 단기적인 이벤트나 일시적인 시장 상황에 의해 형성된 유형은 지속성이 없으므로 장기적인 평가 대상으로 삼지 않습니다.
- 지속성이 확보되면 해당 유형이 앞으로도 계속 존재할 것으로 기대할 수 있습니다.

정리하면,
- 충분성: 데이터가 충분히 존재한다.
- 비교성: 동일 유형 내에서 서로 비교 가능하다.
- 지속성: 유형이 장기적으로 유지될 수 있다.

이 세 조건이 모두 만족될 때 개별 대상이 의미 있는 유형으로 생성됩니다.
