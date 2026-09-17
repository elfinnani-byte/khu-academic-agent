# 대학교 학사 안내 고객응대 에이전트

대학교 학사 규정(수강·등록, 학적, 장학·등록금, 생활) 문의에 답하는 고객응대 에이전트입니다. 문의를 카테고리로 분류(라우팅) → 실제 규정 원문에서 근거를 찾아 답변 생성(그라운딩) → 답변 속 숫자가 근거에 실제로 있는지 검증(가드레일)하는 LangGraph 파이프라인으로 동작하며, 근거가 없으면 지어내지 않고 담당 부서로 이관합니다.

자세한 설계 배경·평가 결과·실패 분석은 [REPORT.md](REPORT.md)를 참고하세요.

## 빠른 시작

```bash
pip install -r requirements.txt
cp .env.example .env   # .env에 OPENAI_API_KEY 입력
python app.py          # http://127.0.0.1:7861
```

`python evaluate.py` 로 터미널에서 바로 평가 지표(라우팅 정확도·macro F1·범위밖 인식률, 1턴 답변 통과율)만 확인할 수도 있습니다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `router.py` | 문의 분류(5개 라우트) + 확신도 기반 처리/이관 판정 |
| `prompts.py` | 라우팅 분류 기준(`ROUTE_GUIDE`)·답변 생성 규칙(`ANSWER_RULES`) 프롬프트 원문 |
| `context.py` | `guide_docs/*.md` 규정 원문을 카테고리별로 조립 |
| `tools.py` / `answer.py` | 조문 검색·장학금/등록금 조회 도구 + 답변 생성 |
| `guardrail.py` | 답변 속 숫자의 근거를 역추적해 미확인 숫자를 걸러냄 |
| `agent.py` | route → answer → guard → escalate 전체 LangGraph |
| `evaluate.py` | 라우팅·답변 성능 채점 |
| `app.py` | Gradio 데모 — 상담 데모 · 성능 벤치마크 · 개선 기록 · 라우트 및 업무 규정 |

## 데이터

`data/`에 라우팅 평가셋 100건(eval/fewshot/outscope), 함정 사례 30건, 답변 채점용 골든셋 16건이 있습니다. `guide_docs/`는 그라운딩에 쓰이는 규정 원문(마크다운) 6종이고, `experiment_log.csv`는 회차별 개선 실험 기록입니다.

## 참고

`.env`에는 실제 API 키가 들어가므로 커밋하지 마세요(`.gitignore`에 이미 포함되어 있습니다). `.env.example`을 복사해서 사용하세요.
