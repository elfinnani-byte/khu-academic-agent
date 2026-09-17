# -*- coding: utf-8 -*-
"""한곳에 모아 둔 설정. 여기 값만 바꿔도 동작이 달라진다.

- MODEL        : 라우팅용. gpt-4.1-nano — 분류는 단순한 작업이라 가장 싼 모델로 충분하다.
- ANSWER_MODEL : 답변 생성용. gpt-4.1-mini — 매뉴얼 컨텍스트가 붙고 지켜야 할 조건이 많아
                 라우팅보다 한 단계 위 모델을 쓴다.
- CONF_THRESHOLD: 이 값 미만이면 사람에게 넘긴다.
- GRADIO_PORT  : 어제 실습(모두몰) 데모가 기본 포트 7860을 쓰므로 겹치지 않게 분리한다.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env의 OPENAI_API_KEY 등을 프로세스 환경변수로 로드

BASE = Path(__file__).parent / "data"

MODEL = os.environ.get("AO_MODEL", "gpt-4.1-nano")
ANSWER_MODEL = os.environ.get("AO_ANSWER_MODEL", "gpt-4.1-mini")

CONF_THRESHOLD = 0.5          # 라우팅 확신도 임계값
MAX_TOOL_TURNS = 3            # 도구 호출 루프 상한
GUARDRAIL_RETRY = 1           # 가드레일 위반 시 재생성 횟수
WORKERS = 8                   # 동시 호출 수. 요청 한도에 걸리면 낮춘다
GRADIO_PORT = 7861            # 어제 실습(모두몰)의 기본 포트 7860과 겹치지 않게 분리

ROUTES = ["ENROLL_REG", "ACADEMIC_STATUS", "SCHOLARSHIP", "STUDENT_LIFE", "OTHER"]
LABELS4 = ["ENROLL_REG", "ACADEMIC_STATUS", "SCHOLARSHIP", "STUDENT_LIFE"]

# 답변 프롬프트에 노출할 사람이 읽는 카테고리 이름 — 영문 코드(ENROLL_REG 등)를 그대로 보여주면
# 모델이 답변 속 근거 표시에 그 코드를 그대로 인용해버리는 문제가 있어 따로 둔다.
ROUTE_LABEL_KO = {
    "ENROLL_REG": "수강·등록",
    "ACADEMIC_STATUS": "학적",
    "SCHOLARSHIP": "장학·등록금",
    "STUDENT_LIFE": "생활",
    "OTHER": "범위밖",
}
