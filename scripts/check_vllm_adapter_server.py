#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def require_ok(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    return response.json()


def collect_output_text(response_payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def collect_chat_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return (message.get("content") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check vLLM LoRA adapter server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="gpt-oss-20b-smoke")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    models_payload = require_ok(requests.get(f"{base_url}/v1/models", timeout=30))

    available_models = [item.get("id", "") for item in models_payload.get("data", [])]
    if args.model not in available_models:
        raise SystemExit(
            f"requested model not found in /v1/models: {args.model}. available={available_models}"
        )

    responses_request_payload = {
        "model": args.model,
        "input": [
            {
                "role": "developer",
                "content": "당신은 제로인 방법론에 근거해 간결하게 답변하는 도메인 어시스턴트입니다.",
            },
            {
                "role": "user",
                "content": "펀드 평가에서 유형분류가 왜 중요한가요?",
            },
        ],
        "reasoning": {"exclude": True},
        "max_output_tokens": 128,
    }

    chat_request_payload = {
        "model": args.model,
        "messages": [
            {
                "role": "developer",
                "content": "당신은 제로인 방법론에 근거해 간결하게 답변하는 도메인 어시스턴트입니다.",
            },
            {
                "role": "user",
                "content": "펀드 평가에서 유형분류가 왜 중요한가요?",
            },
        ],
        "max_tokens": 128,
        "temperature": 0,
    }

    responses_api: dict[str, Any] | None = None
    responses_api_error: str | None = None
    responses_response = requests.post(
        f"{base_url}/v1/responses", json=responses_request_payload, timeout=120
    )
    if responses_response.ok:
        responses_api = responses_response.json()
    else:
        responses_api_error = (
            f"{responses_response.status_code} {responses_response.text.strip()}"
        )

    chat_api = require_ok(
        requests.post(f"{base_url}/v1/chat/completions", json=chat_request_payload, timeout=120)
    )

    result = {
        "models": models_payload,
        "responses_api": responses_api,
        "responses_api_error": responses_api_error,
        "chat_completions_api": chat_api,
        "assistant_text": collect_chat_text(chat_api) or (
            collect_output_text(responses_api) if responses_api else ""
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
