#!/usr/bin/env python3
from __future__ import annotations

"""GPT-OSS 20B용 LoRA 스모크 학습 스크립트.

이 스크립트는 아주 짧은 SFT(supervised fine-tuning) 러닝을 한 번 수행해
"데이터 로드 -> 베이스 모델 로드 -> LoRA 어댑터 학습 -> 저장 -> 재로딩 ->
샘플 추론" 흐름이 실제 환경에서 끝까지 도는지 검증한다.

현재 구현은 전체 모델을 풀 파인튜닝하지 않고 `peft`의 LoRA 어댑터만 학습한다.
즉, 거대한 베이스 가중치는 거의 그대로 두고 일부 선형 계층에 저랭크 보정 행렬만
붙이는 방식을 택했다. 이는 20B급 모델에서 메모리 사용량과 학습 시간을 크게 줄이는
대신, "어댑터가 정상적으로 붙고 저장/재적용되는가"를 확인하는 스모크 테스트 목적에
더 잘 맞는다.
"""

import argparse
import json
from pathlib import Path
from typing import Any

# `torch`는 텐서 연산, dtype/device 제어, 메모리 정리, 추론 모드 관리에 사용된다.
import torch
# `datasets.Dataset`은 Python list를 Trainer가 바로 소비할 수 있는 HF 데이터셋 형식으로
# 감싸 주므로, TRL/Transformers 학습 파이프라인과 자연스럽게 연결된다.
from datasets import Dataset
# `peft`는 전체 가중치 미세조정 대신 LoRA 어댑터만 학습/저장/재적용하기 위해 사용한다.
from peft import LoraConfig, PeftModel, TaskType
# `transformers`는 베이스 Causal LM과 토크나이저 로딩, 채팅 템플릿 적용을 담당한다.
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
# `trl`의 SFTTrainer/SFTConfig는 "텍스트 한 필드를 읽어 next-token prediction 방식으로
# 지도 미세조정"하는 흐름을 간단히 구성해 준다.
from trl import SFTConfig, SFTTrainer

FINAL_CHANNEL_PREFIX = "<|channel|>final<|message|>"
RETURN_TOKEN = "<|return|>"


def read_json(path: Path) -> dict[str, Any]:
    """설정 파일처럼 단일 JSON 객체를 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """학습용 JSONL을 레코드 단위로 읽는다.

    스모크 테스트는 대개 작은 데이터셋을 쓰므로 파일 전체를 메모리로 읽는 단순한
    방식을 택했다. 대용량 스트리밍보다 구현이 단순해, 빠르게 파이프라인 이상 유무를
    확인하는 목적에 적합하다.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def load_train_dataset(path: Path) -> Dataset:
    """TRL 학습기에 넘길 학습 데이터셋을 준비한다.

    현재 스크립트는 각 레코드에 이미 학습용 문자열이 완성되어 있다고 가정하고
    `rendered_text` 필드만 검증한다. 즉, 학습 중에 메시지 목록을 다시 조합하는 대신,
    외부 전처리 단계에서 렌더링한 텍스트를 그대로 쓰는 방식을 선택했다.
    """
    records = read_jsonl_records(path)
    for index, record in enumerate(records, start=1):
        text = record.get("rendered_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {index}: missing rendered_text")
    return Dataset.from_list(records)


def load_prompt_records(path: Path, limit: int) -> list[dict[str, Any]]:
    """추론 샘플에 사용할 프롬프트 레코드를 일부만 읽는다."""
    if path.suffix == ".jsonl":
        records = read_jsonl_records(path)
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"prompt source must be a JSON array: {path}")

    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"record {index}: missing messages for prompt source")
    return records[:limit]


def load_tokenizer(model_name: str, cache_dir: Path):
    """베이스 모델과 짝이 맞는 토크나이저를 로드한다.

    AutoTokenizer를 쓰는 이유는 모델별 토크나이저 클래스가 달라도 공통 인터페이스로
    불러올 수 있기 때문이다. GPT-OSS 계열처럼 커스텀 구현을 둘 수 있는 모델은
    `trust_remote_code=True`가 필요할 수 있다.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        # 모델 저장소가 제공하는 커스텀 tokenizer/modeling 코드를 신뢰하고 실행한다.
        # 범용 안전 옵션은 아니지만, 해당 모델 구현이 AutoClass 기본 구현만으로는
        # 재현되지 않는 경우 필요한 선택이다.
        trust_remote_code=True,
    )
    # 학습/패딩 시 토큰을 오른쪽에 붙인다. Causal LM에서 좌/우 패딩은 배치 정렬과
    # attention mask 해석에 영향을 줄 수 있으므로 명시적으로 고정해 둔다.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        # 많은 Causal LM은 별도 PAD 토큰이 없다. 이 경우 EOS를 PAD로 재사용하면
        # 배치 생성/패딩은 가능해지고, 새 PAD 토큰을 추가해 임베딩 크기를 바꾸는
        # 수고를 피할 수 있다.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def resolve_target_modules(model: AutoModelForCausalLM, requested: list[str]) -> list[str]:
    """LoRA를 실제로 주입할 수 있는 모듈 suffix만 남긴다.

    설정 파일에는 `q_proj`, `v_proj` 같은 축약 이름을 적는 경우가 많다. 모델 내부의
    전체 경로(`model.layers.0.self_attn.q_proj`)와 정확히 일치시키기보다 suffix 기준으로
    비교해, 아키텍처가 조금 달라도 공통 패턴으로 LoRA 대상을 찾는 방식을 사용한다.
    """
    available_suffixes = {name.split(".")[-1] for name, _ in model.named_modules()}
    resolved = [name for name in requested if name in available_suffixes]
    if not resolved:
        raise RuntimeError(
            f"none of the requested target modules were found: {requested}. "
            f"available suffix sample: {sorted(list(available_suffixes))[:30]}"
        )
    return resolved


def load_base_model(model_name: str, cache_dir: Path) -> AutoModelForCausalLM:
    """학습/추론에 사용할 베이스 Causal LM을 로드한다.

    AutoModelForCausalLM은 "다음 토큰 예측" 헤드가 달린 언어모델을 자동 선택한다.
    채팅 모델이라도 내부적으로는 대개 causal LM 형태로 학습/생성하므로 이 클래스를 쓴다.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        # GPT-OSS 저장소가 커스텀 모델 클래스를 제공할 수 있으므로 활성화한다.
        trust_remote_code=True,
        # bfloat16은 fp16보다 overflow에 조금 더 강하면서 메모리를 줄일 수 있어,
        # 지원 하드웨어에서는 20B급 모델 스모크 러닝에 흔히 선택된다.
        dtype=torch.bfloat16,
        # 가중치를 더 메모리 친화적으로 적재해 CPU RAM 피크를 줄인다.
        low_cpu_mem_usage=True,
    )
    # 학습 시 KV cache는 메모리를 많이 차지하고 gradient checkpointing과도 충돌할 수 있어
    # 꺼 둔다. 반대로 순수 추론 최적화 스크립트라면 보통 True를 더 선호한다.
    model.config.use_cache = False
    return model


def build_prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    """메시지 목록을 `assistant final` 직전까지 렌더링한다.

    `apply_chat_template`를 쓰는 이유는 모델마다 system/user/assistant 포맷,
    BOS/EOS 배치, special token 규칙이 다르기 때문이다. 수동 문자열 결합보다
    모델 저장소가 정의한 채팅 포맷을 그대로 따르는 편이 안전하다.
    """
    rendered = tokenizer.apply_chat_template(
        messages,
        # 여기서는 먼저 문자열을 만든 뒤 아래에서 일반 tokenizer 호출로 tensor화한다.
        # 즉시 tokenize=True를 쓰는 방식도 가능하지만, 중간 문자열을 확인하기 쉬운
        # 현재 흐름이 디버깅에는 유리하다.
        tokenize=False,
        # assistant 차례를 여는 generation prompt를 붙여, 모델이 이어서 답변 생성하도록 한다.
        add_generation_prompt=True,
    )
    # gpt-oss는 assistant 턴 안에서 analysis/final/commentary 채널을 직접 고를 수 있다.
    # 사용자 응답만 검증하고 싶을 때는 final 채널 시작을 명시적으로 이어 붙여
    # analysis가 먼저 열리는 확률을 낮춘다.
    return rendered + FINAL_CHANNEL_PREFIX


def build_prompt_inputs(tokenizer, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    """렌더링된 프롬프트를 tokenizer 입력 텐서로 변환한다."""
    prompt_text = build_prompt_text(tokenizer, messages)
    return tokenizer(prompt_text, return_tensors="pt")


def extract_final_text(decoded_text: str) -> str:
    """생성 결과에서 final 채널 본문만 잘라낸다.

    decode 결과에 채널 토큰이나 다음 메시지 시작 토큰이 섞일 수 있으므로
    사용자에게 보여줄 final 텍스트만 보수적으로 추출한다.
    """
    text = decoded_text

    if FINAL_CHANNEL_PREFIX in text:
        text = text.split(FINAL_CHANNEL_PREFIX, 1)[1]

    if RETURN_TOKEN in text:
        text = text.split(RETURN_TOKEN, 1)[0]

    if "<|start|>" in text:
        text = text.split("<|start|>", 1)[0]

    # final 본문 뒤에 다음 채널 토큰이 이어지는 경우를 잘라낸다.
    if "<|channel|>" in text:
        text = text.split("<|channel|>", 1)[0]

    return text.strip()


def run_inference(
    model: PeftModel,
    tokenizer,
    prompt_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """학습 후 저장한 LoRA 어댑터를 다시 불러 샘플 응답을 생성한다.

    여기서의 목적은 품질 평가가 아니라 "어댑터 재로딩과 생성 경로가 정상 동작하는가"
    확인하는 것이다. 따라서 소수 샘플만, 결정론적 설정으로 간단히 생성한다.
    """
    results: list[dict[str, str]] = []
    model.eval()

    for record in prompt_records:
        # 정답 assistant 메시지는 제외하고, 학습된 모델이 이어서 답하도록 입력만 남긴다.
        prompt_messages = [msg for msg in record["messages"] if msg["role"] != "assistant"]
        encoded = build_prompt_inputs(tokenizer, prompt_messages)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # 스모크 테스트이므로 짧은 상한만 둔다. 너무 길면 속도와 메모리 부담이 커진다.
                max_new_tokens=128,
                # 샘플링을 끄면 입력/시드가 같을 때 결과가 더 재현 가능해져 smoke 검증에 유리하다.
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0, input_ids.shape[-1] :]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        final_text = extract_final_text(raw_text)
        user_prompt = next(msg["content"] for msg in prompt_messages if msg["role"] == "user")
        results.append(
            {
                "id": record.get("id", ""),
                "prompt": user_prompt,
                "generation_raw": raw_text.strip(),
                "generation_final": final_text,
            }
        )

    return results


def main() -> None:
    """설정 파일 기준으로 스모크 학습과 샘플 추론을 끝까지 실행한다."""
    parser = argparse.ArgumentParser(description="Run GPT-OSS 20B LoRA smoke training.")
    parser.add_argument(
        "--config",
        default="/home/work/dev_data/fine-tuning/configs/gpt_oss_20b_smoke.json",
        help="Path to JSON config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = read_json(config_path)

    cache_dir = Path(cfg["cache_dir"]).resolve()
    dataset_path = Path(cfg["dataset_path"]).resolve()
    prompt_source_path = Path(cfg["prompt_source_path"]).resolve()
    output_dir = Path(cfg["output_dir"]).resolve()
    report_path = Path(cfg["report_path"]).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["model_name"], cache_dir)
    train_dataset = load_train_dataset(dataset_path)
    prompt_records = load_prompt_records(prompt_source_path, int(cfg["sample_prompt_count"]))

    base_model = load_base_model(cfg["model_name"], cache_dir)
    if hasattr(base_model, "enable_input_require_grads"):
        # 일부 모델/PEFT 조합은 입력 임베딩 경로에 gradient가 흐르도록 이 호출이 필요하다.
        # 모든 모델에서 필수는 아니므로 메서드 존재 여부를 확인한 뒤 안전하게 호출한다.
        base_model.enable_input_require_grads()

    resolved_target_modules = resolve_target_modules(base_model, list(cfg["target_modules"]))
    init_adapter_path = str(cfg.get("init_adapter_path", "")).strip()
    if init_adapter_path:
        init_adapter_dir = Path(init_adapter_path).resolve()
        if not init_adapter_dir.is_dir():
            raise FileNotFoundError(f"init_adapter_path not found: {init_adapter_dir}")
        train_model = PeftModel.from_pretrained(
            base_model,
            str(init_adapter_dir),
            is_trainable=True,
        )
        peft_config = None
    else:
        train_model = base_model
        peft_config = LoraConfig(
            # 현재 코드는 seq2seq나 분류가 아니라 causal language modeling 작업을 택한다.
            task_type=TaskType.CAUSAL_LM,
            # `r`은 LoRA 저랭크 행렬의 rank다. 클수록 표현력은 늘지만 파라미터/메모리도 증가한다.
            r=int(cfg["lora_r"]),
            # `lora_alpha`는 LoRA 업데이트 스케일링 계수다. 실질 업데이트 크기에 영향을 준다.
            lora_alpha=int(cfg["lora_alpha"]),
            # LoRA 경로에만 dropout을 적용해 과적합을 줄인다.
            lora_dropout=float(cfg["lora_dropout"]),
            # 전체 모델이 아니라 attention/projection 계층 일부만 선택적으로 어댑터를 주입한다.
            target_modules=resolved_target_modules,
            # bias까지 학습하지 않고 LoRA 가중치만 학습해 스모크 실험을 더 가볍게 유지한다.
            bias="none",
        )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        do_train=True,
        # bf16 혼합정밀 학습을 사용해 메모리 부담을 줄인다.
        bf16=True,
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        # 작은 per-device batch를 여러 step 누적해 더 큰 유효 배치를 흉내 낸다.
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        # epoch 기준보다 빠르게 끝나는 smoke run 제어를 위해 max_steps를 직접 쓴다.
        max_steps=int(cfg["max_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        logging_steps=int(cfg["logging_steps"]),
        logging_first_step=True,
        # 스모크 테스트는 중간 체크포인트 몇 개면 충분하므로 step 단위 저장만 유지한다.
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=1,
        # 외부 로거(wandb 등) 의존성을 피하려고 비활성화한다.
        report_to="none",
        # 각 샘플에서 학습에 사용할 문자열 필드를 명시한다.
        dataset_text_field="rendered_text",
        # 샘플별 토큰 길이 상한. 너무 길면 메모리 사용량이 급증한다.
        max_length=int(cfg["max_length"]),
        # 여러 짧은 샘플을 이어붙여 packing하지 않는다. 디버깅은 단순하지만 효율은 덜할 수 있다.
        packing=False,
        # activation을 다시 계산해 메모리를 아끼는 대신 속도를 일부 희생한다.
        gradient_checkpointing=True,
        # TRL이 데이터셋 열을 정리하는 과정에서 필요한 원본 필드를 지우지 않도록 둔다.
        remove_unused_columns=False,
        seed=int(cfg["seed"]),
    )

    trainer = SFTTrainer(
        # SFTTrainer는 일반 Trainer보다 텍스트 SFT 파이프라인 구성이 단순하다.
        # 이미 준비된 텍스트 필드와 tokenizer/peft 설정만 넘기면 학습 루프를 구성해 준다.
        model=train_model,
        args=sft_config,
        train_dataset=train_dataset,
        # 최신 TRL 인터페이스에서는 tokenizer 역할을 하는 processing_class를 받는다.
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    # 최종 저장물은 "베이스 모델 전체"가 아니라 LoRA 어댑터 중심 결과물이다.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    train_metrics = dict(train_result.metrics)
    del trainer
    del base_model
    # 대형 모델을 곧바로 다시 읽어야 하므로, 가능한 GPU 캐시를 먼저 비운다.
    torch.cuda.empty_cache()

    reloaded_base = load_base_model(cfg["model_name"], cache_dir)
    # 방금 저장한 LoRA 어댑터를 베이스 모델 위에 다시 얹어 실제 재사용 경로를 검증한다.
    reloaded_model = PeftModel.from_pretrained(reloaded_base, str(output_dir))
    inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reloaded_model = reloaded_model.to(inference_device)
    infer_results = run_inference(reloaded_model, tokenizer, prompt_records)

    report = {
        "status": "success",
        "model_name": cfg["model_name"],
        "init_adapter_path": init_adapter_path or None,
        "dataset_path": str(dataset_path),
        "prompt_source_path": str(prompt_source_path),
        "output_dir": str(output_dir),
        "resolved_target_modules": resolved_target_modules,
        "train_metrics": train_metrics,
        "sample_inference": infer_results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
