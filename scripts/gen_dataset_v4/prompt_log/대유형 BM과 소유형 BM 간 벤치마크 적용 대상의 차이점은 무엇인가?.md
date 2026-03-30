# 질문 프롬프트 로그

## Stage
questions

## Record ID
01_유형분류_기준_structure_0007

## Question
대유형 BM과 소유형 BM 간 벤치마크 적용 대상의 차이점은 무엇인가?

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
  "generated_question": "대유형 BM과 소유형 BM 간 벤치마크 적용 대상의 차이점은 무엇인가?",
  "qa_type": "comparison",
  "question_template": "대유형_vs_소유형_BM_차이_template"
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
v4_q_0018

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
  "question": "대유형 BM과 소유형 BM 간 벤치마크 적용 대상의 차이점은 무엇인가?",
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
대유형 BM과 소유형 BM 간 벤치마크 적용 대상의 차이점은 무엇인가?
```

---

## Assistant Response
```text
- **소유형 BM(적용 대상)**: 각 **소유형**별로 지정된 벤치마크가 적용됩니다. 국내·해외 모두 “대유형-소유형” 체계 하에서 **소유형에 대응하는 ‘소유형 BM’**이 제시됩니다.   
- **대유형 BM(적용 대상)**: “소유형 BM” 중 일부는 **대유형 BM과 동일하게 적용**됩니다. 문서에서는 이때 **해당 소유형의 BM(볼드체)이 ‘해당 소유형이 속한 대유형의 BM과 동일’**하다고 명시합니다.   

즉, 벤치마크 적용은 원칙적으로 **소유형 기준**으로 하되, **특정 소유형은 소유형 BM이 대유형 BM을 그대로 쓰는 방식**으로 구분됩니다.
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
