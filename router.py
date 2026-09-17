# -*- coding: utf-8 -*-
"""의도 분류 라우터. State 하나와 노드 둘로 된 그래프다.

classify 는 모델이 하는 일(분류), gate 는 정책이 정하는 일(처리/이관/범위밖)이다.
둘을 나눠 둔 덕에 임계값만 바꿀 때 모델을 다시 부르지 않아도 된다.

어제 실습(모두몰) router.py와 동일한 구조. 카테고리만 5개(ENROLL_REG/ACADEMIC_STATUS/
SCHOLARSHIP/STUDENT_LIFE/OTHER)로 바뀌었다.
"""
import re
from typing import Literal, Optional, TypedDict

import pandas as pd
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import BASE, CONF_THRESHOLD, MODEL
from prompts import ROUTE_GUIDE


class RouterState(TypedDict, total=False):
    """그래프를 통과하며 채워지는 값들. 노드는 자기가 바꾼 키만 돌려준다."""

    question: str          # 입력 — 학생 문의 한 문장
    route: str              # ① 분류 노드가 채운다
    confidence: float       # ① 분류 노드가 채운다
    reason: str              # ① 분류 노드가 채운다
    action: str              # ② 판정 노드가 채운다 — HANDLE / ESCALATE / OUT_OF_SCOPE
    message: Optional[str]  # ② 판정 노드가 채운다 — 학생에게 바로 나갈 문구


# 규칙 기반 분류 — 기준선. 순서대로 검사해 먼저 걸리는 규칙을 채택한다.
# 학적(전공배정 포함)을 먼저 보고, 그 다음 장학, 그 다음 수강·등록, 마지막 생활 순서.
RULES = [
    (r"휴학|복학|전과|자퇴|제적|재입학|편입학|유급|전공\s?배정|전공\s?이수|다전공|부전공|"
     r"연계전공|학생설계전공|전공\s?신청", "ACADEMIC_STATUS"),
    (r"장학|고시장학|국가유공자|보훈|근로장학|우정장학", "SCHOLARSHIP"),
    (r"등록금|등록\s?기간|등록\s?마감|수강신청|재수강|수강\s?철회|계절학기|계절수업|"
     r"성적|학점|졸업|평점|출석|시험|부정행위", "ENROLL_REG"),
    (r"학생증|동아리|집회|행사|홍보물|학생회|표창|징계|정학|근신", "STUDENT_LIFE"),
]

CONF_THRESHOLD = CONF_THRESHOLD


def classify(state: RouterState) -> RouterState:
    """노드 ① 분류 — 키워드 규칙 버전. LLM 버전으로 갈아 끼우기 전 기준선."""
    q = state["question"]
    for pattern, route in RULES:
        if re.search(pattern, q):
            return {"route": route, "confidence": 0.8, "reason": "키워드 규칙 매치"}
    return {"route": "OTHER", "confidence": 0.3, "reason": "매치되는 규칙 없음"}


def gate(state: RouterState) -> RouterState:
    """노드 ② 판정 — 확신도와 응대 범위를 보고 처리/이관/범위밖을 정한다."""
    if state["confidence"] < CONF_THRESHOLD:
        return {"action": "ESCALATE",
                "message": "정확한 확인을 위해 학사지원팀으로 문의해 주시기 바랍니다."}
    if state["route"] == "OTHER":
        return {"action": "OUT_OF_SCOPE",
                "message": "해당 문의는 이 안내로 답변드리기 어려운 부분입니다. 학사지원팀으로 문의해 주시기 바랍니다."}
    return {"action": "HANDLE", "message": None}


def build(node=None):
    """노드를 이름으로 찾아 그래프를 만든다.

    같은 이름의 함수를 다시 정의한 뒤 build()를 한 번 더 부르면 그 노드만 갈아 끼워진다.
    """
    g = StateGraph(RouterState)
    g.add_node("classify", node or globals()["classify"])
    g.add_node("gate", globals()["gate"])
    g.add_edge(START, "classify")
    g.add_edge("classify", "gate")
    g.add_edge("gate", END)
    return g.compile()


class RouteDecision(BaseModel):
    """학생 문의 한 건에 대한 라우팅 판단 결과."""

    route: Literal["ENROLL_REG", "ACADEMIC_STATUS", "SCHOLARSHIP", "STUDENT_LIFE", "OTHER"] = Field(
        description="문의를 배정할 라우트. 5개 값 중 하나만 사용한다.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="판단의 확신도. 두 라우트 사이에서 애매하면 0.5 미만으로 낮춘다.")
    reason: str = Field(
        description="그 라우트로 판단한 근거를 한 문장으로. 학생이 실제로 알고 싶어하는 결과를 기준으로 쓴다.")


_router_chain = None
_fewshot_block = None


def build_router_chain(model=None, api_key=None):
    """분류용 모델을 만든다."""
    model = model or MODEL
    extra = {"api_key": api_key} if api_key else {}
    return init_chat_model(
        model, temperature=0, timeout=60, max_retries=2, **extra
    ).with_structured_output(RouteDecision)


def _load_fewshot_block():
    """routing_answers.csv의 split=fewshot 문항을 예시로 묶는다(OTHER는 outscope라 제외)."""
    inq = pd.read_csv(BASE / "student_inquiries.csv")
    ans = pd.read_csv(BASE / "routing_answers.csv")
    fs = inq.merge(ans, on="qa_id")
    fs = fs[fs["split"] == "fewshot"]
    lines = [f'- "{r.question}" → {r.route}' for r in fs.itertuples()]
    return "다음은 정답이 확인된 예시입니다:\n" + "\n".join(lines)


def llm_classify(state):
    """분류 노드 — LLM 버전. 이것이 기본값이다."""
    global _router_chain, _fewshot_block
    if _router_chain is None:
        _router_chain = build_router_chain()
    if _fewshot_block is None:
        _fewshot_block = _load_fewshot_block()
    system = ROUTE_GUIDE + "\n\n" + _fewshot_block
    d = _router_chain.invoke(
        [("system", system), ("human", f"학생 문의: {state['question']}")])
    return {"route": d.route, "confidence": d.confidence, "reason": d.reason}


rule_classify = classify          # 규칙 기반 버전을 이름 붙여 남겨 둔다
classify = llm_classify           # 기본 라우터는 LLM
app = build()


def route(question):
    """문의 한 줄을 라우터에 통과시킨다."""
    return app.invoke({"question": question})
