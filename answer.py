# -*- coding: utf-8 -*-
"""도구 호출 루프(agent ↔ tools). 카테고리 근거를 프롬프트에 넣고, 모델이 필요한 도구를 스스로 부르게 한다."""
import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

import tools as _tools
from config import ANSWER_MODEL, MAX_TOOL_TURNS, ROUTE_LABEL_KO
from context import build_context
from prompts import ANSWER_RULES


@tool
def search_regulation(category: str, keyword: str) -> dict:
    """카테고리별 조문 조각에서 키워드로 관련 조문을 검색한다."""
    return _tools.search_regulation(category, keyword)


@tool
def get_scholarship_info(name: str) -> dict:
    """장학금 이름으로 종류·자격기준·지급액을 조회한다."""
    return _tools.get_scholarship_info(name)


@tool
def get_tuition_refund_rate(days_elapsed: int) -> dict:
    """학기 개시일로부터 경과일수로 등록금 반환 비율을 조회한다."""
    return _tools.get_tuition_refund_rate(days_elapsed)


TOOLS = [search_regulation, get_scholarship_info, get_tuition_refund_rate]
TOOL_MAP = {t.name: t for t in TOOLS}

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = init_chat_model(ANSWER_MODEL, temperature=0, timeout=60, max_retries=2).bind_tools(TOOLS)
    return _model


def answer_with_tools(question: str, route: str):
    """문의 한 줄을 답변 루프에 통과시킨다. (답변 텍스트, 호출된 도구 결과 dict) 반환."""
    model = _get_model()
    context = build_context(route)
    system = ANSWER_RULES.format(route_label=ROUTE_LABEL_KO.get(route, route), context=context)
    messages = [("system", system), ("human", question)]

    results = {}
    ai = None
    for _ in range(MAX_TOOL_TURNS):
        ai = model.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content, results
        for tc in ai.tool_calls:
            out = TOOL_MAP[tc["name"]].invoke(tc["args"])
            results[tc["name"]] = out
            messages.append(ToolMessage(content=json.dumps(out, ensure_ascii=False), tool_call_id=tc["id"]))
    return ai.content, results
