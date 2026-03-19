#!/usr/bin/env python3
"""01_upload_doc.py — only_text.md를 OpenAI Files API에 업로드하고 file_id를 저장한다."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_dotenv_file() -> None:
    for dotenv_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"').replace("\r", "")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv_file()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload source document to OpenAI Files API."
    )
    parser.add_argument(
        "--doc",
        default=str(
            Path(__file__).resolve().parents[2]
            / "docs/제로인방법론/Zeroin 펀드평가 방법론 - only_text.md"
        ),
        help="Source markdown document path.",
    )
    parser.add_argument(
        "--state-dir",
        default=str(Path(__file__).resolve().parent / "state"),
        help="Directory to store state files.",
    )
    parser.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    parser.add_argument("--force", action="store_true", help="Re-upload even if already uploaded.")
    return parser.parse_args()


def require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return key.strip().replace("\r", "")


def upload_file(api_base: str, api_key: str, doc_path: Path) -> dict:
    import urllib.request
    api_base = api_base.strip().replace("\r", "")
    api_key = api_key.strip().replace("\r", "")

    file_content = doc_path.read_bytes()
    filename = doc_path.name
    boundary = uuid.uuid4().hex

    def part_field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    def part_file(name: str, fname: str, content: bytes) -> bytes:
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{fname}"\r\n'
            f"Content-Type: text/markdown\r\n\r\n"
        ).encode("utf-8")
        return header + content + b"\r\n"

    body = (
        part_field("purpose", "user_data")
        + part_file("file", filename, file_content)
        + f"--{boundary}--\r\n".encode("utf-8")
    )

    req = urllib.request.Request(
        url=f"{api_base}/files",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    args = parse_args()
    api_key = require_api_key()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "upload_state.json"

    if state_path.exists() and not args.force:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"already uploaded: file_id={state['file_id']} (use --force to re-upload)")
        return

    doc_path = Path(args.doc).resolve()
    if not doc_path.exists():
        raise FileNotFoundError(f"document not found: {doc_path}")

    print(f"uploading: {doc_path.name} ({doc_path.stat().st_size:,} bytes) ...")
    result = upload_file(args.api_base, api_key, doc_path)

    file_id = result["id"]
    state = {
        "file_id": file_id,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "filename": doc_path.name,
        "bytes": result.get("bytes", doc_path.stat().st_size),
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"uploaded: file_id={file_id}")
    print(f"state saved: {state_path}")


if __name__ == "__main__":
    main()
