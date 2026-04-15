# Context Log

이 문서는 작업 중 나온 배경, 원인, 의도, 제약, 결정 이유를 누적 기록하는 용도입니다.
기존 내용을 수정하거나 삭제하지 않고, 새로운 항목을 아래에 계속 추가합니다.

## 2026-03-10

### 설정
- 워크스페이스 전체에 항상 적용되는 기록 규칙을 사용한다.
- 사용자가 어떤 식으로 설명하더라도 배경, 원인, 의도, 제약, 결정 이유로 해석될 수 있으면 기록한다.
- 기록 대상 파일은 `docs/context-log.md`로 고정한다.

### 메모
- 기록은 원문 그대로 복붙하기보다 짧고 구조화된 요약으로 남긴다.

## 2026-03-10

### 큰그림
- 허깅페이스 계열 모델인 `qwen3.5-27b`와 `gpt-oss-120b`를 대상으로 답변 품질을 비교하려고 한다.
- 학습 대상 문서는 `docs/제로인방법론/Zeroin 펀드평가 방법론.md`이다.
- 문서 내용을 기반으로 모델을 로컬 학습한 뒤 `vllm`으로 서빙해서 응답 결과를 비교할 계획이다.

### 의도
- 동일한 도메인 문서를 학습한 두 모델의 답변 결과를 비교 평가하려고 한다.

### 학습 방식
- 여기서 말한 학습은 `RAG`나 단순 프롬프트 주입이 아니라 `LoRA` 파인튜닝을 의미한다.

## 2026-03-10

### 데이터셋 생성 계획
- `datasets/canonical/qa_seed_poc_v1/seed_ch01.jsonl`와 `datasets/canonical/qa_seed_poc_v1/seed_ch02.jsonl`를 초기 seed 데이터로 사용한다.
- 이 seed를 기반으로 `ai-api`를 사용해 질문-답변 데이터셋을 추가 생성할 계획이다.
- 초반에는 각 단원별로 약 10개 수준의 데이터를 먼저 생성해 소규모로 학습해본다.

### 실험 운영 방식
- 소규모 데이터 생성 후 먼저 학습을 진행한다.
- 학습 결과와 테스트 결과는 별도 테스트 문서에 기록한다.
- 이후 동일한 자료를 바탕으로 `ai-api`를 사용해 데이터셋을 추가로 약 50개 더 생성하는 식으로 점진적으로 확장한다.

### 의도
- 처음부터 대량 생성하지 않고, 작은 배치로 학습과 평가를 반복하면서 데이터 품질과 모델 반응을 확인하려고 한다.

## 2026-03-10

### gpt-oss 데이터셋 준비
- `datasets/canonical/qa_seed_poc_v1/seed_ch01.jsonl`를 기반으로 `datasets/gpt-oss/` 아래에 `gpt-oss`용 소규모 샘플 데이터셋을 만들기 시작한다.
- 첫 시도는 JSON 파일 하나에 3개 샘플만 담아 작게 검증한다.

### 포맷 결정
- `gpt-oss` LoRA 미세조정용 데이터는 `messages` 구조를 유지한 채 저장하고, 실제 학습 시 `tokenizer.apply_chat_template()`로 Harmony 포맷으로 렌더링하는 방향으로 간다.

### 현재 상태
- 초기 학습 재료는 준비되었고, 다음 단계는 학습 실행 파이프라인으로 넘어가는 상태다.

### Harmony 렌더링
- `datasets/gpt-oss/seed_ch01_first3.json`의 앞 3개 샘플에 대해 실제 `gpt-oss` tokenizer의 `apply_chat_template()`를 사용한 Harmony 렌더링 결과 파일을 생성했다.
- 렌더링 결과는 `datasets/rendered/gpt-oss/seed_ch01_first3_harmony.jsonl`에 저장한다.

### 렌더링 스크립트
- Harmony 렌더링을 반복 실행할 수 있도록 `datasets/gpt-oss/render_harmony.py` 스크립트를 추가했다.
- 이 스크립트는 canonical `messages` 입력을 받아 `rendered_text` JSONL로 변환한다.

## 2026-03-11

### llm_datasets 구조 변경 대응
- 데이터셋 구조가 `llm_datasets/seed_v1`, `llm_datasets/rendered/gpt-oss`, `llm_datasets/render_gpt-oss-harmony.py` 기준으로 재배치되었다.
- 렌더링 스크립트의 기본 출력 경로와 `source_file` 기록 방식을 새 구조 기준으로 수정했다.
- 수정 후 `seed_v1/seed_ch01_first3.json` 입력으로 렌더링 파일을 재생성해 경로 메타가 올바르게 반영되는 것을 확인했다.

## 2026-03-11

### 저장소 연결
- 현재 워크스페이스를 비어 있는 GitHub 저장소 `yminstudio/nipa-finetune`에 연결하기로 했다.
- 워크스페이스 등록 시 `@.archive` 디렉터리는 무시 대상으로 유지한다.

### 제약
- `@.archive`는 탐색뿐 아니라 Git 추적 대상에서도 제외하는 방향으로 정리한다.

### 브랜치 기준
- 원격 저장소의 첫 기본 브랜치는 `main`이 아니라 `master` 기준으로 맞추기로 했다.

## 2026-03-11

### gpt-oss-20b 스모크 목표
- 이번 단계의 목표는 `llm_datasets/rendered/gpt-oss/seed_ch01_first3_harmony.jsonl` 3건으로 `gpt-oss-20b` 학습 파이프라인이 실제로 끝까지 동작하는지 검증하는 것이다.
- 전체 Harmony 렌더링 확장은 이번 범위에서 제외하고, 이미 준비된 샘플을 바로 학습 입력으로 사용한다.

### 실행 기준
- 학습 스택은 `Transformers + TRL + PEFT`를 우선 사용한다.
- 성공 기준은 `학습 완료 + adapter 재로딩 + 샘플 질문 추론 1~2개 성공`으로 둔다.

## 2026-03-11

### 커스텀 서브에이전트 요구
- 파일 경로를 입력으로 받아 코드를 검토하고 상세 주석을 추가하는 전용 서브에이전트를 프로젝트에 두기로 했다.
- 이 서브에이전트는 단순 설명보다, 사용된 라이브러리와 클래스/모듈/함수 옵션의 의미와 선택 이유를 자세히 주석으로 설명하는 역할을 맡는다.

## 2026-03-11

### Git 추적 정리 의도
- 워크스페이스 루트의 `.gitignore`에 Python 프로젝트 기준의 기본 제외 항목을 보강해 캐시, 가상환경, 에디터 설정, 로그 파일이 추적되지 않도록 정리한다.

## 2026-03-11

### Git ignore 재정리
- 모델 베이스 디렉터리 `llm_model_base/`는 계속 전체 무시하되, `llm_model_lora/`는 산출물 대부분을 무시하면서 하위 `README.md`만 Git 추적 대상으로 남기도록 규칙을 조정한다.
- 스테이징을 막던 `.git/index.lock`이 실행 중인 Git 없이 남아 있는 상태라면 stale lock으로 보고 제거해 정상적인 `git add`가 다시 가능하도록 정리한다.

## 2026-03-11

### 서빙 단계 확장
- 사용자는 3건 스모크 학습 검증 다음 단계로 서빙까지 진행하기를 원한다.
- 메인 계획 파일 기준 목표는 최종적으로 `vLLM` 서빙이지만, 현재 환경에는 `vllm`이 설치되어 있지 않다.

### 서빙 실행 전략
- 현재 단계는 `merge -> MXFP4` 최종 배포가 아니라, 학습된 LoRA adapter를 `vLLM`에서 먼저 로컬 서빙해 보는 중간 검증 단계로 잡는다.
- 기존 학습 환경의 커스텀 PyTorch와 충돌 위험을 줄이기 위해 `vLLM`은 별도 격리 가상환경에서 설치/실행하는 방향으로 간다.

## 2026-03-11

### Git 로컬 설정 접근성
- 사용자는 워크스페이스에서 `.git`과 설정 파일이 바로 보이지 않아 로컬 Git 작성자 정보 설정을 더 직접적으로 실행할 수 있는 방법을 원한다.
- 저장소 루트에서 바로 실행 가능한 로컬 Git 작성자 정보 설정 스크립트를 제공해 설정 경로를 직접 찾지 않아도 되게 한다.

## 2026-03-11

### 로컬 설정 스크립트 실행 오류
- Git 로컬 작성자 정보 설정 스크립트가 초기 저장 시 `CRLF` 줄바꿈으로 들어가 Linux 셸에서 `bash\r` 오류가 발생했다.
- 스크립트 줄바꿈을 `LF`로 정리해 저장소 터미널에서 직접 실행 가능하도록 수정한다.

## 2026-03-11

### 저장소 전용 Git 작성자 정보
- 이 저장소에서는 전역 설정이 아니라 로컬 Git 설정만 사용한다.
- 로컬 작성자 정보는 이름 `cm.yun`, 메일 `cm.yun@yminstudio.com`으로 맞춘다.

## 2026-03-12

### 현재 진행 상태
- `gpt-oss-20b` 샘플 학습은 완료되었고, 다음 단계는 학습된 결과를 실제 서빙 경로로 검증하는 것이다.

### 테스트 의도
- `vLLM + OpenWebUI` 조합으로 로컬 대화 테스트를 해 보려 한다.
- 우선순위는 학습된 LoRA adapter가 `vLLM`에서 정상 로드되고, 이어서 OpenWebUI에서 호출 가능한지 확인하는 것이다.

### 접속 방식 변경
- 이후 테스트는 로컬만이 아니라 외부 네트워크에서 접속하는 방향으로 고려한다.
- 가능하면 모델 API를 직접 노출하기보다 UI 계층만 외부에 열고, 모델 서버는 내부망에 두는 구성이 더 적합하다.

### OpenWebUI 운영 상태
- 이 서버에는 별도 가상환경 `venv_openwebui` 기반의 `OpenWebUI` 설치본이 이미 있었다.
- 기존 `OpenWebUI` 프로세스는 남아 있었지만 실제 리슨하지 않는 비정상 상태였고, `openwebui-data`를 데이터 디렉터리로 사용해 재시작했다.
- 재시작 후 `OpenWebUI`는 `0.0.0.0:8080`에서 정상 리슨하며 로컬 접속은 성공했다.
- 다만 공개 IP `14.63.187.10:8080`으로의 자기접속 테스트는 timeout이라, 현재 외부 경로에는 방화벽 또는 네트워크 ACL 차단 가능성이 있다.

### 외부 접속 우회 방안
- 공개 IP 경로가 막혀 있을 가능성이 있어, 사용자는 `ngrok`로 외부 접근을 우회하는 방안을 검토한다.
- 현재 로컬 기준 `OpenWebUI`와 `vLLM`은 모두 정상 응답하므로, 공개 대상은 `vLLM`보다 `OpenWebUI` 단일 포트가 더 적합하다.

### ngrok 사용 의도
- 사용자는 `OpenWebUI` 외부 접근을 위해 `ngrok` 터널을 사용하려고 한다.
- 인증 토큰은 별도 설정에 사용하되, 작업 기록에는 값 자체를 남기지 않는다.

### ngrok 재연결 결과
- 사용자가 기존 `ngrok` 엔드포인트를 정리한 뒤, `zai.aidev.ngrok.dev -> localhost:8080` 터널 재연결이 성공했다.
- 공개 URL에서 `OpenWebUI`의 `/api/version` 응답이 정상 확인되어 외부 접근 경로가 살아 있는 상태다.

## 2026-03-12

### 확장 학습 및 서빙 목표
- 사용자는 기존 `seed_ch01_first3.json` 스모크 성공 이후, `llm_datasets/seed_v1/seed_ch01.jsonl`와 `llm_datasets/seed_v1/seed_ch02.jsonl` 전체를 사용한 실제 학습 확장을 원한다.
- 이번 목표는 두 seed 데이터로 `gpt-oss-20b` LoRA 학습을 다시 수행하고, 결과 adapter를 `vLLM`으로 서빙한 뒤 `OpenWebUI` 외부 접근까지 다시 성공시키는 것이다.

### 운영 원칙
- 이번 단계의 배포 검증은 기존과 동일하게 최종 병합본이 아니라 `adapter` 서빙 경로를 우선 사용한다.
- 기존 스모크 산출물은 보존하고, 새 학습/서빙 결과물은 별도 실험 경로로 분리한다.

## 2026-03-16

### Seed v2 스킬 요구
- 사용자는 `docs/제로인방법론/divide`의 문서들을 `llm_datasets/seed_v2` 학습 데이터로 만들기 위한 프로젝트 스킬 생성을 원한다.
- 출력 형식은 `seed_v1`과 유사한 `messages` 중심 JSONL을 기준으로 하며, 기본 샘플 단위는 섹션 중심 + 필요 시 문서 전체 요약 샘플 추가 방식으로 잡는다.
- 스킬에는 작성 절차, 스키마/템플릿, 그리고 문서 inventory 생성 및 JSONL 검증용 보조 스크립트를 포함한다.

### Seed v2 coverage 보강
- 사용자는 상위 제목 Q/A가 하위 제목 전체를 포괄하도록 스킬 프롬프트를 강화하길 원한다.
- `##`는 하위 `###`, `####` 전체를, `###`는 하위 `####` 전체를 설명할 수 있어야 하며, 표가 있으면 표 해설용 데이터셋 샘플도 별도로 생성해야 한다.

### Seed v2 생성 범위 확정
- 사용자는 `docs/제로인방법론/divide` 아래 각 Markdown 문서를 문서 전체 기준으로 누락 없이 학습 데이터화하길 원한다.
- 결과물은 통합 파일이 아니라 문서별 JSONL 파일로 각각 생성하며, 파일명은 원본 문서명을 반영한 `seed_v2_<문서명>.jsonl` 형식을 사용한다.

### Seed v2 우선순위 재정의
- 사용자는 `출처/파일명 언급 금지`, `문서에 없는 기준 생성 금지` 같은 제약보다 `##, ###, #### 제목이 하위 내용을 포함해 요약되는가`를 더 중요한 원칙으로 본다.
- 따라서 제목 계층 전체를 포함한 요약을 방해하는 제약은 생성 스크립트와 스킬 프롬프트에서 제거하고, 상위 제목이 하위 제목 전체를 포괄하는 생성 규칙을 최우선으로 둔다.

### Seed v2 생성 엔진 전환
- 사용자는 내부 `vLLM` 기반 재작성 품질과 안정성이 부족하다고 판단해, 외부 고성능 AI API를 사용한 생성 방식으로 전환하길 원한다.
- 외부 API는 `OpenAI API`를 기준으로 하며, 인증 정보는 저장소 루트 `.env`의 환경변수 형식으로 관리한다.

## 2026-03-18

### Seed v4 설계 방향
- `v4`는 `scripts/gen_dataset_v4`와 `llm_datasets/seed_v4`를 대상으로 설계한다.
- `v3`는 현재 설계 범위에서 제외하고, 나중에 비교 검토 대상으로만 둔다.
- 범위는 `1~2단원`으로 제한한다.

### v4 생성 원칙
- 사용자가 키워드를 수작업으로 넣지 않는다.
- 질문 재료는 `단원 제목 + 절/소절 제목 + 본문 핵심 명사`를 사용한다.
- 질문 생성과 답변 생성은 분리된 2단계 파이프라인으로 설계한다.
- 답변은 업로드한 문서 근거 안에서만 생성하고, 장황한 일반론과 출처 암시 표현은 억제한다.

### v4 출력 방향
- 최종 산출물은 `messages + meta` 구조의 JSONL로 저장한다.
- 초기 출력은 단원별 파일 분리 구조를 기본으로 한다.
- 외부 API 기본 모델은 `gpt-5-nano`를 사용한다.

### Chapter 02 재작성 재개 기준
- 사용자는 `docs/제로인방법론/divide/02_펀드평가_방법론.md`를 대상으로 `llm_datasets/seed_v2/seed_v2_02_펀드평가_방법론.jsonl`의 OpenAI 기반 재작성을 이어서 완료하길 원한다.
- 가장 중요한 품질 기준은 heading coverage이며, `##`는 하위 `###`, `####`를, `###`는 하위 `####`를 빠짐없이 요약해야 한다.
- 이번 단계에서는 canonical JSONL의 재작성, 검증, 품질 점검까지만 진행하고 Harmony 렌더링이나 학습은 chapter 02 품질이 충분히 확인된 뒤에만 진행한다.

### Seed v2 Harmony 렌더링
- 사용자는 `seed_v2` 문서별 canonical JSONL을 `gpt-oss` 학습용 Harmony 형식으로 변환하길 원한다.
- `llm_datasets/seed_v2/` 아래 12개 파일을 모두 `llm_datasets/rendered/gpt-oss/seed_v2_*_harmony.jsonl`로 렌더링했다.

### Chapter 03 이후 주요 장 재작성
- 사용자는 chapter 02 이후의 주요 장 7개(`03`, `04`, `05`, `06`, `07`, `08`, `X1`)를 원문 Markdown 기준으로 다시 정비해 canonical `seed_v2` JSONL 품질을 끌어올리길 원한다.
- 재작성은 로컬 `vLLM`이 아니라 저장소 루트 `.env`의 설정을 읽는 외부 `OpenAI API` 경로를 사용하며, 기본 모델은 `gpt-5-nano`다.
- 최우선 품질 기준은 heading coverage로, `##`는 하위 `###`와 `####`를, `###`는 하위 `####`를 빠짐없이 포괄해야 한다.
- 기존 파일은 가능하면 이어받되, 품질이 약하면 보존보다 전면 재작성 품질을 우선하고, 각 장은 재작성 후 즉시 JSONL 검증까지 완료한다.
- 이번 단계에서는 canonical JSONL 품질 확보와 검증까지만 진행하며, 대상 파일 전체가 충분히 정리되기 전에는 Harmony 재렌더링을 진행하지 않는다.

### Seed v2 전체 Harmony 재렌더링
- 이후 `seed_v2` canonical JSONL 전체(`00`, `01`, `02`, `03`, `04`, `05`, `06`, `07`, `08`, `X1`, `목차`, `개정사항`)를 다시 `gpt-oss` Harmony 형식으로 일괄 렌더링했다.

### Seed v2 통합 학습 입력 준비
- 사용자는 이제 `seed_v2` 전체를 기준으로 학습 단계까지 진행하길 원한다.
- 문서별 canonical JSONL 12개와 Harmony JSONL 12개를 각각 `seed_v2_all.jsonl`, `seed_v2_all_harmony.jsonl`로 합쳐 통합 학습 입력을 만들고, 이를 사용하는 새 학습 config를 추가한다.

### GPT-OSS 계획 진행 방식
- 사용자는 `gpt-oss-20b` 모델 선택 계획을 실제 실행 단계로 이어서 진행하길 원한다.
- 이후 작업에서는 `서브에이전트`와 프로젝트 `스킬` 정의도 함께 보강해, `seed_v2 -> Harmony -> LoRA 학습 -> 병합 -> 직접 검증 -> vLLM 서빙` 흐름을 더 적극적으로 재사용할 수 있게 하길 원한다.

### Adapter 우선 운영 전환
- 사용자는 현 단계에서 `LoRA` 품질 확인보다 먼저 병합까지 진행하는 것은 원하지 않는다.
- 따라서 당분간 실사용 검증 경로는 `병합본`이 아니라 `adapter + vLLM + OpenWebUI` 조합으로 유지한다.
- `OpenWebUI`에서 베이스 모델과 LoRA 모델이 함께 보이는 것은 정상으로 보고, 실제 선택 대상은 `gpt-oss-20b-seed-v2-all` adapter 모델로 맞춘다.

### 2차 학습안 연결
- 사용자는 현재 1차 결과의 품질 실패를 바탕으로 `2차 학습안`까지 실제 실행 가능한 형태로 이어서 정리하길 원한다.
- 2차는 `seed_v2_all`을 유지하되, `max_length`, `gradient_accumulation`, `max_steps`, `attention-only LoRA 범위`, `검증 질문셋`을 강화한 별도 round2 config로 분리한다.
- 2차 검증도 병합보다 먼저 `adapter + vLLM + OpenWebUI` 경로에서 진행한다.

### Round2 실행 결과
- `configs/gpt_oss_20b_seed_v2_all_round2.json` 기준 2차 학습은 완료됐다.
- 학습 자체는 안정적으로 수렴했지만, 직접 검증 결과는 여전히 제로인 원문 규칙보다 일반 금융 상식 수준 답변으로 평탄화되는 경향이 남아 있었다.
- 일부 샘플에서는 `주제`, `포맷`, `제약` 같은 메타 문구를 그대로 답하는 현상도 확인돼, 다음 단계는 추가 학습보다 canonical 데이터 정비가 우선이라는 판단을 세웠다.

## 2026-03-17

### seed_v2.1 직접 검증 및 vLLM 서빙
- adapter 직접 검증 완료: 7개 validation_questions에 대해 생성 테스트, 리포트 `llm_model_lora/gpt-oss-20b-seed-v2-1/adapter_validation.json` 저장.
- adapter 검증 시 병합 모델 경로 대신 베이스 모델을 사용하도록 `check_gpt_oss_model_output.py` 수정.
- vLLM adapter 서버 기동: 포트 8001에서 `gpt-oss-20b-seed-v2-1` adapter 서빙 중. (포트 8000은 기존 seed-v2-all 사용 중)

### seed_v2.1 학습 실행
- 사용자가 seed_v2.1 데이터 품질을 검토 후 학습 실행을 요청했다.
- `configs/gpt_oss_20b_seed_v2_1.json` 생성, Harmony 렌더링(155건), LoRA 학습을 순차 실행했다.
- GPU 2가 vLLM 등으로 점유되어 `CUDA_VISIBLE_DEVICES=0,1`로 학습을 실행했다.
- Round2와 동일한 하이퍼파라미터(max_length 1024, max_steps 240, lora_r 16 등)를 적용했다.

### seed_v2.1 수동 보정 후보 추출
- `01_유형분류_기준.jsonl`, `02_펀드평가_방법론.jsonl`에서 품질 이슈가 있는 레코드 10건을 추출해 `llm_datasets/seed_v2_1/manual_edit_candidates.md`에 정리했다.
- 우선 1(8건): 질문/답변 에코, `주제에 따라`, `표의 구성과 용어를 실제 값 중심으로 정리` 등 메타 표현.
- 우선 2(2건): placeholder 문구(`이어집니다`, `확장된다` 등).

## 2026-03-19

### v4 adapter OpenWebUI 연결
- 사용자는 `feature-gen-dataset-v4` 워크트리의 LoRA adapter `gpt-oss-20b-seed-v4-all-round1`를 `OpenWebUI`에서 선택해 확인하려고 한다.
- 초기에는 `OpenWebUI`가 `127.0.0.1:8000/v1`를 바라보고 있었지만, 해당 포트의 `vLLM` 서버가 꺼져 있어 모델 목록이 비어 있었다.
- 루트 저장소 기준 분리 환경 `.venv-vllm`을 다시 준비한 뒤, 워크트리 adapter 경로를 사용해 `vLLM` adapter 서버를 `8000` 포트에 기동했다.
- 확인 결과 `/v1/models`에 `gpt-oss-20b-seed-v4-all-round1`가 노출되고, `chat/completions` 호출도 정상 응답했다.

### OpenWebUI 원격 검증 진행 방식
- 사용자는 원격 환경이라 에이전트가 직접 `OpenWebUI` 로그인과 모델 검증까지 수행하길 원한다.
- 로그인 자격 정보는 대화 중 제공되었으며, 작업에는 사용하되 기록 파일에는 민감한 값을 남기지 않는다.

### v4 답변 길이 선호
- 사용자는 `gen_dataset_v4` 답변 생성에서 다소 길어지더라도 정보 전달의 난이도를 낮추는 쪽을 선호한다.

### v4 개선 정리 범위
- 사용자는 이번 단계에서 장기 전략보다 `데이터셋 보정과 추가 샘플 설계`에 집중하길 원한다.
- 따라서 우선 실패 원인 유형화, 보정 규칙, 우선 추가 샘플 초안까지 정리하는 방향으로 진행한다.

### v4 프롬프트 중심 운영 의도
- 사용자는 `질문 생성`, `답변 생성` 품질을 주로 프롬프트로 해결하길 원한다.
- 원본문서는 업로드 파일로 제공하고, 질문 리스트 생성과 업로드 파일 기반 답변 생성 모두 프롬프트 계약을 강화하는 방향을 선호한다.
- 필터나 후처리 코드는 최소화하고, 보정 역시 가능하면 프롬프트 안에서 해결되길 원한다.

## 2026-03-25

### p6 프롬프트 규칙 도입
- 사용자는 답변 품질의 핵심을 길이 제어가 아니라 `무엇을 남기고 무엇을 버릴지`를 강제하는 데 있다고 본다.
- 새 공통 규칙 `p6`: 문서에 없는 내용은 추가하지 않되, 의미 연결을 위한 최소한의 설명만 허용한다.
- `p6`의 허용 범위는 용어 풀이와 생략된 주어/관계 보완처럼 문맥을 잇는 수준이며, 새로운 판단 기준, 규칙, 예외, 질문에 없는 대상/비교, 일반 금융 지식 확장은 금지한다.
- `p6`에는 답변을 자연스러운 문장으로 연결하되, 새로운 정보 추가 없이 기존 내용을 매끄럽게 표현해야 한다는 원칙도 포함된다.
- 사용자는 이 규칙을 프로젝트 전반 프롬프트에 공통 적용하길 원한다.
- 이후 피드백으로 `p6`만으로는 부족하고, `definition`, `criteria`, `comparison`, `application`별로 무엇을 반드시 남기고 무엇을 제외할지 더 강한 타입별 hard rule이 필요하다는 점이 확인되었다.

### p6 기반 round 학습 스킬 재구성
- 사용자는 `p6` 철학을 반영한 프롬프트 상태를 바탕으로 `seed_round+1` 데이터 생성, 학습, 그리고 철학 적합성 검토까지 이어지는 프로젝트 전역 스킬을 원한다.
- 기존 워크트리 내부에 있던 `zeroin-training-validation` 성격의 스킬이 있으면 이를 루트 프로젝트 스킬로 재구성해 공유 가능한 형태로 두길 원한다.
- 기본 운영 철학은 `adapter-only`, `round+1` 기본, 직접 검증 필수, `p6` 기준 검토 필수이다.

## 2026-03-29

### v5 allchap round2 파이프라인 완료 및 검증
- 워크트리 `feature-gen-dataset-v5`에서 `seed_v5_allchap_round2_all.jsonl` 808건, Harmony 렌더 후 `gpt_oss_20b_seed_v5_all_round2.json` 설정으로 round1 어댑터 이어서 학습 완료(`train_loss` 약 0.636, 240 steps).
- 학습 후 `check_gpt_oss_model_output.py` 기본 `--question-limit` 5로는 설정의 `validation_questions` 8개 중 일부만 평가되므로, 동일 어댑터에 대해 `--question-limit 8`으로 전체 검증을 재실행함.
- 8문항 직접 검증에서 `p6` 관점의 주요 이슈: 일부 응답이 방법론 고정이 아니라 일반 금융·회계 상식으로 채워짐(예: 회전율을 매출액/평균자산으로 설명), 수식·수치·구간의 비현실적 나열·반복, 동일 문장을 항목별로 복붙한 듯한 출력, 메타 자기비판 문구 혼입 등. 즉 학습 파이프라인은 끝까지 통과했으나, 배포 전 품질 게이트로 문서 대조·RAG·추론 시스템 프롬프트 보강 등이 여전히 필요하다는 판단.

### p6 재정의 반영: 독립적 이해 가능성 강화
- 사용자는 `p6`가 단순한 단답형·압축형이 아니라, 답변이 독립적으로 이해 가능하도록 질문 대상과 기본 맥락을 아주 짧게 복원해야 한다고 본다.
- 따라서 `무확장` 원칙은 유지하되, `질문의 대상을 짧게 다시 잡아 주는 최소 전제 설명`을 허용하는 방향으로 답변 생성 프롬프트와 직접 검증 프롬프트를 함께 보강한다.

### v5 allchap round3 실행 요청
- 사용자는 워크트리 `feature-gen-dataset-v5`에서 최근 p6 프롬프트 수정이 반영된 상태를 기준으로 다음 라운드인 `round3` 전체 파이프라인 실행을 요청했다.
- 범위는 `seed 생성 -> Harmony 생성 -> config 정합성 확인/필요 시 보정 -> LoRA 학습 -> 직접 adapter 검증 -> p6 기준 리뷰`이며, 가능하면 기존 `10_run_v5_all_chapters.py` 오케스트레이터 흐름을 우선 활용한다.
- 이번 라운드는 `round2` 완료본을 재사용하되 새 round 산출물을 만들어야 하며, 직접 검증은 `check_gpt_oss_model_output.py`로 설정의 검증 질문 전체 개수를 반영해야 한다.
- 핵심 품질 기준은 `문서 기반 + 최소 연결 + 무확장`을 유지하면서도, 단답형 압축이 아니라 `질문의 대상을 짧게 다시 잡아 주는 최소 맥락 허용`이 실제 출력에 반영되는지 확인하는 것이다.

### p6 형식 지시 축소
- 사용자는 `3~5개 불릿`, `2~4문장`처럼 답변 형식과 개수를 직접 유도하는 문구가 p6 철학과 맞지 않는다고 판단했다.
- 따라서 답변 생성 프롬프트에서는 개수/형식 허용 문구를 제거하고, 대신 `무엇을 남길지`와 `무엇으로 확장하지 말지`만 남기는 방향을 유지한다.

### p6 길이 제한 제거
- 사용자는 `답변 길이는 공백 포함 최대 N자` 같은 길이 제한도 p6 철학과 맞지 않는다고 판단했다.
- 따라서 답변 생성 프롬프트와 결과 메타데이터에서 길이 상한 문구와 `answer_max_chars` 기록을 제거하고, 길이보다 내용 선택 원칙을 우선한다.

### 시스템 지시와 사용자 프롬프트의 역할 분리
- 사용자는 `예시, 장황한 배경설명, 불필요한 열거는 줄인다` 같은 일반 제약은 시스템 레벨에서 이미 충분히 요청된 사항이므로, 사용자 프롬프트에 중복으로 둘 필요가 없다고 판단했다.
- 따라서 답변 생성용 사용자 프롬프트에서는 중복 제약 문구를 제거하고, 사용자 프롬프트에는 질문별로 직접 필요한 지시만 남긴다.

## 2026-03-30

### v7 단일 Q/A 반복 학습 전환
- 사용자는 `v6`의 API/검증 중심 흐름 대신, 새 브랜치 `v7`에서 질문 1개와 답변 1개만으로 반복 학습하는 가장 단순한 로컬 경로를 원한다.
- 따라서 `v7`에서는 외부 API 호출, judge, 후속 smoke 생성 검증을 기본 실행 경로에서 제거하고 `train_result.json` 중심으로만 학습 완료를 확인한다.

## 2026-04-01

### v7 round5 표 기반 확장 학습 완료
- 사용자는 `scripts/data_source.md`를 기준으로 기존 `round4` 다음 라운드를 만들고, 표의 각 행을 answer group으로, 각 `<br>` 질문 변형을 개별 학습 레코드로 확장한 뒤 베이스 모델에서 새 라운드를 학습하고 전체 질문 smoke 로그까지 남기길 원했다.
- 따라서 `scripts/build_v7_round_from_data_source.py`를 추가하고, 현재 표 기준 5개 그룹·150개 질문으로 `seed_v7_single_qa_round5.jsonl`, `gpt_oss_20b_seed_v7_single_qa_round5.json`, `smoke_v7_single_qa_round5_config.json`를 생성했다.
- `round5` 학습은 베이스 모델 `unsloth/gpt-oss-20b-BF16`에서 GPU 0으로 수행했고, `loss < 1.0` 및 `mean_token_accuracy >= 0.98` 조건으로 `step 115`, `loss 0.1100`, `mean_token_accuracy 0.9804`에서 조기 종료되었다.
- 전체 150문항 smoke 추론은 GPU 2에서 수행해 `tests/log/v7_single_qa_round5_all_questions_report.md`를 생성했다. 출력은 대체로 원문 구조를 유지했지만, 일부 후반 문항에서 일반화된 재서술, 용어 흔들림, 간헐적 표현 오류(`제로인 설정`, `비교 대상가 설정`)가 보여 후속 품질 점검 포인트가 남았다.

## 2026-04-02

### v8 새 워크트리 초기화
- 사용자는 다음 라운드로 이어 가지 않고, `v7`를 기반으로 한 새 워크트리 `v8`를 별도로 준비하길 원했다.
- 따라서 `feature-gen-dataset-v8`는 `feature-gen-dataset-v7` 브랜치를 출발점으로 새 git worktree로 생성하고, 이전 round의 `llm_model_lora/` 산출물 흔적과 `tests/log/` smoke 로그를 제거한 깨끗한 실험 시작점으로 정리한다.

### v8 풀 파인튜닝 전환 설계
- 사용자는 `v8`부터는 LoRA 어댑터가 아니라 `gpt-oss-20b` 전체 가중치를 업데이트하는 풀 파인튜닝 경로로 전환하길 원했다.
- 따라서 `v8` 설계는 기존 `data_source -> dataset -> config -> train -> report` 흐름은 유지하되, 학습 엔트리를 `full_ft` 전용 스크립트와 설정으로 분리하고 산출물은 `llm_model_full/` 아래에 저장한다.
- 실제 `v8 round1` 실행은 하드웨어와 Python 패키지 스택 readiness gate를 먼저 통과한 경우에만 진행하고, 검증도 adapter reload 대신 저장된 전체 모델 또는 체크포인트 재로딩 기반으로 바꾼다.

### v8 round1 데이터 소스 확정
- 사용자는 `v8` 학습 입력으로 `/home/work/dev_data/fine-tuning/.worktrees/feature-gen-dataset-v8/scripts/data_source.md`를 사용할 계획이라고 밝혔다.
- 따라서 `v8 round1` 설계는 이 Markdown 표를 기준 소스로 삼고, 각 행의 `assistant` 답변과 `<br>`로 나뉜 `user` 질문 변형을 JSONL 학습 레코드로 확장하는 전제를 명시한다.

### v8 풀 파인튜닝 전환 배경 보강
- 사용자는 `v7`에서 LoRA 기반 학습을 여러 차례 진행했지만 원하는 수준의 결과가 충분히 나오지 않았다고 판단했다.
- 따라서 `v8`에서는 더 강한 학습 경로가 필요하다고 보고, LoRA 대신 `gpt-oss-20b` 전체 가중치를 업데이트하는 풀 파인튜닝을 시도한다.
- 이 전환 전에 `H200 x3` 환경에서 `gpt-oss-20b` 풀 파인튜닝이 가능한지 검토했고, 조사 결과 실행 가능하다고 판단되어 `v8` full fine-tuning 설계를 진행한다.

### v8 Task 2 구현 범위 고정
- 사용자는 격리된 `feature-gen-dataset-v8` worktree에서 계획 문서의 Task 2만 구현하길 요청했다.
- 따라서 `scripts/data_source.md`를 읽어 `llm_datasets/seed_v8/seed_v8_round1_full_ft.jsonl`을 생성하는 최소 builder, 해당 경로를 검증하는 테스트, 그리고 `<br>` 질문 확장·결정론적 id·고정 시스템 프롬프트·`prompt_source_path` 재사용 전제를 우선 구현한다.
- 이번 작업에서는 git commit을 만들지 않고, Task 3 이후 범위나 worktree 밖 파일은 건드리지 않으며, red/green 테스트와 builder CLI 직접 실행으로만 검증한다.

### v8 운영 문서 최소 업데이트
- 사용자는 Task 7 범위에서 `v7`와 `v8`의 차이, `v8 full_ft` 데이터셋·산출물 위치, dry-run/실학습/사후 검증 실행법을 운영자가 바로 이해할 수 있게 문서를 최소 수정하길 원했다.
- 따라서 `docs/v7_single_qa_training.md`에는 `v8 full_ft` 전환 차이를 짧게 연결하고, `docs/v8_preparation.md`에는 `build dataset -> dry-run -> real training -> post-training validation` 순서와 핵심 경로를 간단한 운영 가이드로 정리한다.

### v8 H200 x3 첫 실학습 착수
- 사용자는 `/.venv-gpt-oss-train`에 `deepspeed`를 설치하고, GPU 1의 `vLLM` 프로세스를 내려 `H200 x3`로 실제 `v8 round1` 풀 파인튜닝을 시작하길 원했다.
- 따라서 학습용 가상환경에 `deepspeed`를 설치하고 기존 `vLLM` 점유를 해제한 뒤, `torchrun --nproc_per_node=3` 기준 readiness/dry-run을 재확인하고 실학습을 시작했다.
- 첫 런타임 검증에서 `Trainer(tokenizer=...)` 호환성 문제와 DeepSpeed `gradient_accumulation_steps` 불일치가 드러나 이를 즉시 수정했고, 중복 실행으로 생긴 OOM 상황을 정리한 후 수정된 설정으로 단일 clean run을 다시 시작했다.

### v8 Harmony 재설계 선행 원칙 확인
- 사용자는 `gpt-oss` 출력 누수 문제를 고치기 전에, 먼저 변경 설계를 파일로 남기는 기존 작업 규칙을 다시 지키길 원했다.
- 따라서 `v8 full_ft`의 다음 수정은 바로 코드부터 건드리지 않고, 학습 문자열 렌더링과 검증 출력 파싱을 Harmony 기준으로 재정의한 설계 문서를 먼저 작성한 뒤 사용자 검토를 거쳐 진행한다.

## 2026-04-06

### v8 Harmony 기준 재학습 결정
- Harmony 기준 학습 문자열 렌더링과 `final` 채널 추출 로직을 코드와 회귀 테스트에 반영한 뒤, 사용자는 이제 `v8 round1`을 해당 기준으로 다시 학습하길 원했다.
- 따라서 이번 재실행은 기존 복구 export를 그대로 신뢰하지 않고, 같은 `seed_v8_round1_full_ft.jsonl`과 `gpt_oss_20b_seed_v8_round1_full_ft.json` 설정을 사용하되 Harmony 정렬이 반영된 최신 러너 기준으로 처음부터 다시 학습한다.
- 재학습 후에는 저장된 `final-export`를 다시 로드해 새 validator로 raw 출력과 사용자용 `final` 답변을 함께 검증한다.

### 디스크 확보를 위한 이전 산출물 백업 삭제 승인
- 재학습 시작 직전 readiness gate가 `disk_free_bytes` 부족으로 차단되었고, 직전에 보관한 이전 `v8` full-ft 백업이 약 `587G`를 차지하고 있었다.
- 사용자는 이 백업을 삭제해 공간을 확보하고, 같은 설정으로 Harmony 기준 재학습을 바로 다시 시작하는 방안을 선택했다.

### v8 디스크 readiness gate 기준 재조정
- 실제 파일시스템 가용 공간은 약 `1.1 TiB`였고, 직전 `v8` full-ft 백업 전체 크기는 약 `587G`였지만, 러너는 `1.5 TiB` 최소 여유를 강제해 재학습을 시작조차 하지 못했다.
- 따라서 이번 단계에서는 실측 산출물 규모를 기준으로 readiness gate의 최소 디스크 요구치를 `1 TiB`로 낮추고, 회귀 테스트로 `1 TiB` 환경이 통과해야 한다는 계약을 추가했다.

### v8 재시도와 서브에이전트 모니터링 전환
- 조정된 readiness gate 기준으로 재학습을 다시 시작했지만, 이번에는 내부 예외 없이 `45 step` 부근에서 `torchrun` 부모 프로세스가 외부 `SIGTERM`을 받아 중단되었다.
- 사용자는 같은 설정으로 다시 재시도하되, 이후 상태 확인은 서브에이전트 기반 모니터링으로 넘기고 완료 시점에만 결과를 보고받는 방식을 원했다.

### v8 checkpoint 기반 품질 확인 전환
- 이후 재시도에서는 `1000 step`까지 학습과 `checkpoint-1000`/`final-export` 생성이 진행됐지만, 마지막 분산 종료 구간에서 `NCCL collective timeout`으로 런이 실패 처리되었다.
- 사용자는 다음 단계로 실제 출력 품질을 확인하길 원했고, 현재 `final-export`는 비어 있으므로 검증 기준 산출물은 `checkpoint-1000`으로 잡아 reload validation을 수행한다.

### v8 final export NCCL timeout 수정 및 clean rerun 착수
- 사용자는 이번 단계에서 `final export` 저장/종료 구간의 `NCCL timeout`을 먼저 수정한 뒤, 같은 학습 하이퍼파라미터로 다시 실행해 실제 `success` 런을 확보하길 원했다.
- 따라서 `final export` 단계는 writer-only phase가 아니라 모든 rank가 참여하는 collective phase로 바꾸고, writer만 tokenizer 부가 산출물을 쓰도록 조정했다.
- 이전 실패 산출물은 보존한 채 clean rerun을 위해 동일 설정에 새 `output_dir`만 사용하는 `gpt_oss_20b_seed_v8_round1_full_ft_retry1.json`을 만들고, 이후 런에서 답변 잘림이 남으면 그때 `max_new_tokens`나 종료 조건을 재조정해 다시 확인한다.
- 기존 `checkpoint-1000` 품질 리포트에서 답변이 문장 중간에서 반복적으로 끊기는 패턴이 `max_new_tokens=256` 상한과 맞물려 보였기 때문에, 이번 재검증 기준값은 validator 기본 `max_new_tokens`를 `512`로 높여 새 런 종료 후 잘림 여부를 다시 확인한다.

### v8 retry1 종료 후 export/검증 확인
- `retry1` 산출물에서는 `checkpoint-1000/trainer_state.json`의 `global_step`이 `1000`으로 확인됐고, `final-export` 디렉터리에 `model.safetensors`, tokenizer, config 산출물이 정상 존재했다.
- `final-export`를 대상으로 validator를 다시 실행한 결과 `v8_round1_full_ft_retry1_final_export_report_512.md` 기준 `Run Status: success`였고, Harmony `final` 답변도 정상 추출되었다.
- 직전 `256` 기준에서 보이던 답변 중간 잘림은 `512` 기준 샘플에서는 재현되지 않았지만, 러너가 자체로 남기는 `train_result.json`은 여전히 dry-run 값이라 최종 report 기록 경로는 별도 점검 여지가 남는다.

### v8 retry1 산출물 비교 평가 기록
- 사용자는 `retry1` 산출물을 기존 `checkpoint-1000` 기준 결과와 비교해 설명하고, 그 평가를 문서로 남기길 원했다.
- 비교 결과, 가장 큰 개선은 `final-export` 실산출물 확보와 재로딩 검증 성공, 그리고 기존 `256` 기준에서 보이던 답변 중간 잘림이 `512` 기준 샘플에서는 사라졌다는 점이었다.
- 반면 답변 내용의 기본 골격은 기존과 거의 동일했고 질문별 표현 변화 폭도 크지 않아, 이번 결과는 "내용 수준의 큰 도약"보다는 "운영 완결성 확보와 답변 완결성 개선"으로 해석하는 것이 적절하다고 정리했다.

### v8 round2 착수 전 round1 산출물 정리
- 사용자는 기존 `round1` full fine-tuning 산출물을 유지할 필요가 크지 않다고 판단했고, 비교는 이제 베이스 모델을 기준으로 하면 된다고 정리했다.
- 따라서 `gpt-oss-20b-seed-v8-round1-full-ft`와 `gpt-oss-20b-seed-v8-round1-full-ft-retry1` 산출물을 삭제해 약 `1.1T`의 여유 공간을 회복했다.
- 이후 같은 데이터셋을 유지한 채 출력 경로만 새로 잡는 `gpt_oss_20b_seed_v8_round2_full_ft.json`을 만들었고, `round2` 실학습 전 dry-run이 다시 `ready`로 통과하는 것을 확인했다.

### v8 round2 세션 단절 대응 재실행
- 사용자는 Cursor 세션이 끊겨도 학습이 유지되도록 실행 방식을 바꾸길 원했고, 직전 `round2` 런은 `checkpoint-100`까지만 남긴 채 외부 세션 단절로 중단됐을 가능성이 높다고 판단했다.
- `checkpoint-100`은 resume-capable로 확인되어, `resume_mode=resume_latest`를 사용하는 `gpt_oss_20b_seed_v8_round2_full_ft_resume1.json`을 새로 만들고 dry-run으로 `checkpoint-100` 재개가 가능함을 확인했다.
- 이후 `setsid`로 부모 세션과 분리된 detached 프로세스로 `round2` 이어학습을 다시 시작했고, 전용 로그 파일 `round2-resume1.log`를 기준으로 모니터링 서브에이전트가 상태를 추적하도록 전환했다.

### v8 round2 종료부 writer phase 재설계와 품질 평가
- 사용자는 `round2`의 종료부 `NCCL timeout`을 실제로 수정하고, 확보된 `final-export` 산출물의 품질 평가까지 이어서 진행하길 원했다.
- 조사 결과 `final export` 자체는 성공했고 `.runner-sync/final-export*.json`도 모두 `ok`였으며, 실제 timeout은 그 다음 `reload validation`/`train result report`를 writer-only barrier 구조로 수행하는 동안 non-writer rank가 10분 NCCL 대기 후 실패한 흐름과 맞았다.
- 따라서 이번에는 `final export` 뒤에서 stale 상태 파일을 짧게 정리한 후 process group을 먼저 종료하고, 이후 `reload validation`과 `train result report`는 writer가 상태 파일을 쓰고 non-writer는 파일만 기다리는 post-training writer phase로 분리하도록 러너/테스트를 조정했다.
- 같은 시점에 현재 `round2 final-export`를 별도 validator로 재검증한 결과 reload validation은 `success`였고, 샘플 기준으로 Harmony `final` 추출 정상, 내부 추론 누출 없음, `max_new_tokens=512` 기준 답변 잘림 없음으로 평가했다.

### v8 round3 1단원 CSV 시작점과 데이터 확장 방향
- 사용자는 세션마다 업로드한 라운드용 CSV를 `제로인방법론` 폴더 쪽에 저장해 관리하고 있으며, `round3`는 우선 `v8-r3 section-1.csv` 같은 1단원 데이터셋 파일부터 시작할 계획이라고 설명했다.
- 동시에 디스크 여유가 부족한 상황에서 데이터를 모아서 한 번에 학습해야 하는지 고민하고 있으며, 현재까지의 평가 결과가 나쁘지 않으므로 데이터 확장 중심으로 다음 라운드를 설계하려는 의도가 있다.

### v8 round3 1단원 검증 샘플 다양화 요청
- 사용자는 `round3 section1` 산출물이 직접 검증만 되면 종료부 에러는 핵심이 아니라고 보았고, 기존 자동 샘플이 모두 `충분성·비교성·지속성` 질문 변형이라 확인 가치가 낮다고 지적했다.
- 따라서 `section1` 전체에서 서로 다른 `group_id`를 대표하는 20개 질문을 별도 검증 prompt source로 추려 `final-export`에 다시 적용하고, 사용자가 직접 읽어볼 수 있는 분산 샘플 보고서를 새로 만든다.

### v8 검토 기본 생성 길이 1024 상향
- 사용자는 이후 모델 검토 시 샘플이 중간에서 잘리지 않도록 검토 기본값을 `1024`로 고정하길 원했다.
- 따라서 `check_gpt_oss_full_ft_output.py`의 기본 `max_new_tokens`를 `1024`로 상향하고, 관련 validator 보고서 테스트 기대값도 함께 맞춘다.

### v8 round4 전체 문서 CSV 전환과 디스크 차단
- 사용자는 기존 `v8-r3 section-1.csv` 대신 컬럼 위치가 조정된 `docs/제로인방법론/source_csv/v8-r4 section-all.csv`를 기준으로 문서 전체 데이터셋을 만들고, 그 데이터로 full fine-tuning을 진행하길 원했다.
- 새 CSV는 `num, sec, system, user_q_base, assistant, user_q_ext` 구조로 확인되었고, `round4 section-all` builder/config를 추가해 dataset 생성까지 진행했다.
- 다만 `num=51` 행은 `assistant`가 비어 있어 학습용 Q/A로는 사용할 수 없어 builder에서 해당 행만 제외했고, 최종 dataset은 `394`개 그룹 `4023`개 레코드로 생성되었다.
- 이후 dry-run에서는 현재 가용 디스크가 약 `482G`로 부족해 readiness gate가 차단되었고, 현재 큰 산출물은 `round2`와 `round3 section1` full model 디렉터리가 각각 약 `587G`씩 차지하고 있다.

### v8 round4 출력 파일시스템 분리
- 사용자는 `dev_data` 마운트에서 추가 정리를 계속하되, 실제 학습은 더 여유 있는 다른 파일시스템으로 `output_dir`를 옮겨서 진행하길 원했다.
- 확인 결과 `/home/work`는 `ext4` 마운트이며 약 `2.5T` 여유가 있어, `round4 section-all` full fine-tuning 출력 경로와 `train_result.json` 경로를 `/home/work/llm_model_full/gpt-oss-20b-seed-v8-round4-section-all-full-ft`로 이동한다.
- dataset, prompt source, deepspeed config는 기존 worktree 경로를 유지하고, 대형 체크포인트/최종 export만 별도 파일시스템에 기록하도록 구성해 readiness gate를 우회하지 않고 통과시키는 방향으로 정리한다.

### v8 round4 `/home/work` 출력 경로 기준 실학습 시작
- 변경된 `output_dir` 기준으로 `torchrun --nproc_per_node=3` dry-run을 다시 수행한 결과, readiness는 `ready`였고 `disk_free_bytes`는 약 `2.73T`로 기록되어 디스크 차단이 해소되었다.
- 이후 실제 학습은 세션 단절에 영향받지 않도록 detached 프로세스로 시작했고, 로그는 `/home/work/llm_model_full/gpt-oss-20b-seed-v8-round4-section-all-full-ft/round4-train.log`, PID 파일은 같은 디렉터리의 `round4-train.pid`에 기록한다.
- 초기 로그에서는 HF Hub 비인증 경고와 `torch_dtype` deprecated 경고만 보였고, 별도 training monitor로 초기 진행/실패 여부를 계속 추적한다.

## 2026-04-15: post-training validation 프로세스 분리

- **문제**: round4 학습 1000/1000 완료 후, 러너가 같은 프로세스 안에서 모델을 다시 로드하여 검증하려 했으나, ZeRO-3 분산 학습으로 GPU당 ~129GB가 점유된 상태에서 39GB 모델 재로딩이 불가능하여 매 라운드 post-training 단계에서 hang 발생.
- **근본 원인**: `run_from_config` 내 `coordinated_post_training_writer_phase("reload validation")`이 in-process로 모델을 재로딩하는 구조. 학습 프로세스가 종료되어야 GPU 메모리가 해제되므로, 같은 프로세스에서는 재로딩 불가.
- **수정**: 러너에서 in-process reload validation을 제거하고, `--orchestrate` 모드를 추가.
  - 학습 러너: 학습 + final-export + train_result.json 기록까지만 하고 정상 종료
  - 오케스트레이터: training subprocess(torchrun) 완료 후 → validation subprocess(check_gpt_oss_full_ft_output.py) 자동 실행
  - 프로세스 단위 분리로 GPU 메모리 충돌 원천 차단
- **사용법**: `python run_full_ft_gpt_oss_20b.py --config CONFIG --orchestrate [--nproc-per-node N] [--skip-validation]`
- round4 학습 산출물(final-export 39GB, checkpoint-1000)은 정상 저장 확인됨.

