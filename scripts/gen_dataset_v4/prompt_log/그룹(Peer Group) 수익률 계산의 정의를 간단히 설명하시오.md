
---

## Stage
answers

## Record ID
v4_q_0028

## Model
gpt-5.4-nano

## API Base
https://api.openai.com/v1

## Extra
```json
{
  "vector_store_id": "vs_68a80414c938819189ac784ba37c10ee",
  "chapter": "2. 펀드평가 방법론",
  "section": "2.3. 수익률 평가방법",
  "subsection": "2) 그룹(Peer Group) 수익률 계산",
  "seed_title": "2) 그룹(Peer Group) 수익률 계산",
  "question": "그룹(Peer Group) 수익률 계산의 정의를 간단히 설명하시오.",
  "answer_policy": "definition_compact"
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
그룹(Peer Group) 수익률 계산의 정의를 간단히 설명하시오.
```

---

## Assistant Response
```text
그룹(Peer Group) 수익률은 특정 운용사 유형의 성과(또는 유형 평균 성과)를 측정하기 위해, 해당 유형(그룹)에 속한 펀드들을 하나의 펀드(Composite)로 보고 수익률을 계산하는 개념입니다. 

계산은 (1) 먼저 그룹의 일별 수익률을 산출한 뒤 , (2) 해당 일별 수익률을 기간 동안 시간가중(Time Weighted) 방식으로 누적(곱셈)해 기간 수익률을 구합니다.
```

## Response Extra
```json
{
  "answer_status": "generated",
  "answer_issues": [],
  "answer_policy": "definition_compact",
  "vector_store_id": "vs_68a80414c938819189ac784ba37c10ee"
}
```
