# 질문 프롬프트 로그

## Stage
questions

## Record ID
01_유형분류_기준_structure_0007

## Question
국내펀드의 유형별 BM에서 '대유형 BM'의 정의는 무엇인가?

## Model
gpt-5-nano

## API Base
https://api.openai.com/v1

## Extra
```json
{
  "chapter": "1. 유형분류 기준",
  "section": "1.2 유형별 벤치마크",
  "subsection": "국내펀드의 유형별 BM",
  "seed_title": "국내펀드의 유형별 BM",
  "questions_per_record": 3,
  "generated_question": "국내펀드의 유형별 BM에서 '대유형 BM'의 정의는 무엇인가?",
  "qa_type": "definition",
  "question_template": "대유형BM_정의_template"
}
```

## System Prompt
```text
당신은 제로인 방법론 기반 QA 데이터셋의 질문 생성기다.
반드시 한국어로만 답하고, 제공된 재료 범위 안에서만 질문을 만든다.
질문은 짧고 단일 쟁점 중심이어야 한다.
일반 금융상식, 투자조언, 문서 바깥 배경지식 질문은 금지한다.
업로드 문서를 보지 않고도 일반 금융 상식만으로 답할 수 있는 질문은 생성하지 않는다.
질문은 반드시 문서 안의 특정 정의, 기준, 비교, 적용 중 하나에 직접 매핑되어야 한다.
```

## User Prompt
```text
아래 재료만 사용해서 자연스러운 질문 후보를 생성해줘.

[단원 제목]
1. 유형분류 기준

[절 제목]
1.2 유형별 벤치마크

[소절 제목]
국내펀드의 유형별 BM

[핵심 제목]
국내펀드의 유형별 BM

[핵심 명사]
국내펀드의 유형별 BM, 대유형, 소유형, 소유형 BM, 국내펀드의, 유형별, BM, 주식형, 일반주식, 순수, 자산배분, KOSPI200

규칙:
- 질문은 짧고 단일 쟁점 중심
- 문서 범위를 벗어난 질문 금지
- 제목을 그대로 복붙한 문장 금지
- 일반 금융상식, 투자조언, 시장전망 질문 금지
- 업로드 문서를 보지 않고도 일반 금융 상식만으로 답할 수 있는 질문은 생성하지 않는다
- 답이 표 전체 요약이나 장문 설명이 되도록 만드는 질문은 금지하고, 특정 기준 하나만 묻는 질문으로 만든다
- 질문 하나에는 하나의 판단축만 남긴다. 정의와 기준, 기준과 예외를 한 문장에 함께 묻지 않는다
- 질문 유형은 definition, criteria, comparison, application 중 하나만 사용
- 최대 3개까지만 생성

생성 후 자체 점검:
- 이 질문은 문서의 특정 표, 기준, 정의, 절차가 없으면 답하기 어려운가?
- 이 질문에 일반 금융 상식으로 그럴듯하게 답할 수 있으면 폐기한다
- 질문이 길거나 복합적이면 더 짧은 한 쟁점 질문으로 다시 쓴다

반환 형식:
순수 JSON 배열만 출력한다.
[
  {
    "question": "질문 내용",
    "qa_type": "definition|criteria|comparison|application",
    "question_template": "템플릿 이름"
  }
]

```

---

## Stage
answers

## Record ID
v4_q_0016

## Model
gpt-5.4-nano

## API Base
https://api.openai.com/v1

## Extra
```json
{
  "vector_store_id": "vs_68a80414c938819189ac784ba37c10ee",
  "chapter": "1. 유형분류 기준",
  "section": "1.2 유형별 벤치마크",
  "subsection": "국내펀드의 유형별 BM",
  "seed_title": "국내펀드의 유형별 BM",
  "question": "국내펀드의 유형별 BM에서 '대유형 BM'의 정의는 무엇인가?",
  "answer_policy": "chapter1_summary"
}
```

## System Prompt
```text
제로인은 펀드평가사로 펀드평가의 방법론을 가지고 있습니다.
당신은 제로인 방법론에 근거해 답변하는 도메인 어시스턴트다.
반드시 한국어로만 답한다.
업로드된 문서에 근거가 없는 기준, 수치, 예외를 만들지 않는다.
질문에 필요한 최소 범위까지만 답한다.
문서 밖 일반 지식으로 빈칸을 메우지 않는다.
답변은 짧고 구조적으로 유지한다.
출처/원문/문서/장/절을 직접 언급하지 않는다.
일반 금융 상식으로 확장하지 않는다.
```

## User Prompt
```text
국내펀드의 유형별 BM에서 '대유형 BM'의 정의는 무엇인가?
```

---

## Assistant Response
```text
국내펀드의 유형별 BM에서 **‘대유형 BM’**은 **‘대유형(예: 주식형, 주식혼합형, 채권형 등)’ 단위로 설정되는 기준 벤치마크**를 의미합니다. 해당 대유형 아래의 **소유형**에 대해, 표에 제시된 **소유형 BM 중 볼드 처리된 BM은 “해당 소유형이 속한 대유형의 BM과 동일”**하다고 명시되어 있습니다. 

(예시로, 국내펀드 유형별 BM 표에서 대유형별로 제시되는 대유형 BM은 다음과 같이 나타납니다: 주식형–KOSPI200, 주식혼합형–KOSPI200 50% + 제로인 종합채권지수(1-2Y) 50%, 채권혼합형–KOSPI200 25% + 제로인 종합채권지수(1-2Y) 75%, 채권형–KAP 종합채권 2년 90%, MMF–제로인 MMF 지수, 부동산형–제로인 대안투자기대수익지수 5Y)
```

## Response Extra
```json
{
  "answer_status": "generated",
  "answer_issues": [],
  "answer_policy": "chapter1_summary",
  "vector_store_id": "vs_68a80414c938819189ac784ba37c10ee"
}
```
