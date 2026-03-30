
---

## Stage
answers

## Record ID
v4_q_0026

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
  "subsection": "1) 펀드 수익률 계산",
  "seed_title": "1) 펀드 수익률 계산",
  "question": "펀드 수익률 계산에서 '기준 NAV'와 '실제주식편입비 NAV' 간 차이가 수익률 산출에 어떤 영향을 미치나요?",
  "answer_policy": "comparison_medium"
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
펀드 수익률 계산에서 '기준 NAV'와 '실제주식편입비 NAV' 간 차이가 수익률 산출에 어떤 영향을 미치나요?
```

---

## Assistant Response
```text
- 기준 NAV는 그룹의 BM 수익률을 계산할 때 쓰입니다. 제로인에서는
  - 기준 NAV( GNAV ) = 수정NAV( MDNAV ) × BM 일수익률( R^BM )
  이고, 이를 이용해 BM의 일별 수익률은 “∑기준NAV / ∑수정NAV - 1”로 산출합니다.  

- 실제주식편입비 NAV는 그룹의 “실제주식편입비 수익률”을 계산할 때 쓰입니다. 제로인에서는
  - 실제주식편입비 NAV( SNAV’ ) = 수정NAV( MDNAV ) × 실제편입비 일수익률( R’ )
  이고, 이를 이용해 실제주식편입비 일별 수익률은 “∑실제주식편입비NAV / ∑수정NAV - 1”로 산출합니다.  

- 따라서 “기준 NAV vs 실제주식편입비 NAV의 차이”는 분모(∑수정NAV)가 동일하게 놓인 상태에서, 분자에 들어가는 값이 (BM 일수익률 기반 vs 실제편입비 일수익률 기반)으로 달라지므로 **BM 수익률과 실제편입비 수익률이 달라지고**, 그 결과 **BM 대비 초과수익 및 성과요인분석(자산배분/타이밍/종목선정 효과) 계산의 비교 기준이 달라집니다.**
```

## Response Extra
```json
{
  "answer_status": "generated",
  "answer_issues": [],
  "answer_policy": "comparison_medium",
  "vector_store_id": "vs_68a80414c938819189ac784ba37c10ee"
}
```
