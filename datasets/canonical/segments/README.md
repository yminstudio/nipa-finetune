## Zeroin 방법론 문서 세그먼트 (canonical)

이 폴더는 `dev_data/docs/제로인방법론/`의 마크다운 원문을 **헤딩 구조를 기준으로 세그먼트화**한 결과를 저장합니다.

### 생성 방법

```bash
python dev_data/fine-tuning/scripts/segment_methodology_md.py
```

기본 입력은 `dev_data/docs/제로인방법론/divide/`이며, 기본 출력은 `dev_data/fine-tuning/datasets/canonical/segments/v1/`입니다.

기본값으로 아래 파일들은 **데이터셋 생성에서 제외**합니다.

- `dev_data/docs/제로인방법론/divide/00_서문.md`
- `dev_data/docs/제로인방법론/divide/목차.md`
- `dev_data/docs/제로인방법론/divide/개정사항.md`
- `dev_data/docs/제로인방법론/divide/X1_기타.md`

포함하려면 아래 옵션을 사용합니다.

```bash
python dev_data/fine-tuning/scripts/segment_methodology_md.py --include-preface
```

```bash
python dev_data/fine-tuning/scripts/segment_methodology_md.py --include-appendix
```

```bash
python dev_data/fine-tuning/scripts/segment_methodology_md.py --include-meta
```

### 산출물

- `v1/segments.jsonl`: 세그먼트 JSONL (1줄 = 1 세그먼트)
- `v1/manifest.json`: 입력 문서별 해시/라인수/세그먼트 수 요약
- `dev_data/fine-tuning/datasets/reports/doc_segmentation_v1.md`: 요약 리포트

### `segments.jsonl` 레코드 스키마(요약)

- `segment_id`: 안정적인 세그먼트 ID
- `segment_kind`: `leaf` 또는 `intro` (`intro`는 상위 헤딩의 “자식 헤딩 이전” 도입부)
- `chapter_id`: 파일명 기반 단원 ID (`01`~`08`, `X1` 등)
- `chapter_title`: 단원 제목(첫 헤딩에서 추출)
- `section_path`: `chapter_id>...` 형태의 경로(숫자 섹션 번호 우선, 없으면 제목 anchor)
- `heading_chain`: 루트→현재 헤딩까지의 제목 배열
- `heading_level`: 마크다운 헤딩 레벨(1~6)
- `content_md`: 원문 마크다운(표/수식/코드블록을 분해하지 않고 그대로 포함)
- `source.doc_path`, `source.doc_hash`, `source.line_start`, `source.line_end`: 원문 추적용 메타
- `stats.*`: 길이/표/수식/이미지 포함 여부 등

