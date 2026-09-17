# -*- coding: utf-8 -*-
"""라우터 · 조회 · 생성 · 가드레일을 하나의 그래프로 잇는다.

route → answer → guard → (END | answer 재시도 | escalate)

모두몰 agent.py와 동일한 뼈대. 차이는 두 가지뿐이다.
1) guardrail()이 route를 추가로 받는다(도구 결과 + 카테고리 컨텍스트를 함께 근거로 봄).
2) ASK 판정을 "도구 미호출 여부"가 아니라 "답변이 되묻는 문장인가"로 한다 — 우리는 카테고리
   컨텍스트를 프롬프트에 통째로 주입해서, 도구를 안 불러도 정답을 내는 경우가 흔하기 때문이다.
"""
import operator
import re
from typing import Annotated, List, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from answer import answer_with_tools
from config import GUARDRAIL_RETRY
from guardrail import guardrail
from router import app as router_app
from tools import escalate_to_agent


class AgentState(TypedDict, total=False):
    """오늘 만든 파이프라인 전체가 통과하는 State."""

    question: str
    history: Annotated[list, operator.add]
    route: str
    confidence: float
    action: str                 # HANDLE / ASK / ANSWER / ESCALATE / OUT_OF_SCOPE
    tools: List[str]
    results: dict
    answer: str
    guardrail_ok: Optional[bool]
    attempts: int


_ASK_RE = re.compile(r"\?|주시겠|알려주|말씀해")


def _as_text(v) -> str:
    """UI 쪽에서 문자열이 아니라 리스트/딕트(예: 멀티모달 content)로 넘어와도 안전하게 문자열로 만든다."""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join(_as_text(x) for x in v if x is not None)
    if isinstance(v, dict):
        return _as_text(v.get("text") or v.get("content") or "")
    return str(v)


def with_history(state: AgentState) -> str:
    """현재 질문만 쓴다.

    원래는 앞 턴 발화를 이어붙여 맥락을 유지하려 했으나, 완전히 무관한 새 질문에도 이전 주제가
    섞여 답이 오염되는 문제가 실제로 발생했다(예: 장학금 질문 뒤에 "학생식당 메뉴가 뭔가요?"를
    물었더니 장학금 답변이 그대로 반복됨). 매 턴을 독립적으로 판단하는 쪽이 더 안전하다고 판단해
    되돌렸다.
    """
    return _as_text(state["question"])


def node_route(state: AgentState) -> AgentState:
    """① 분류 — 라우터 그래프를 그대로 부른다."""
    r = router_app.invoke({"question": with_history(state)})
    return {"route": r["route"], "confidence": r["confidence"], "action": r["action"]}


def node_answer(state: AgentState) -> AgentState:
    """② 조회 + ③ 생성 — 안에서 도구 호출 루프가 한 바퀴 돈다."""
    text, results = answer_with_tools(with_history(state), state["route"])
    if not results and _ASK_RE.search(text):
        return {"action": "ASK", "tools": [], "results": {}, "answer": text,
                "history": [state["question"]]}
    return {"tools": list(results), "results": results, "answer": text,
            "history": [state["question"]],
            "attempts": state.get("attempts", 0) + 1}


def node_guard(state: AgentState) -> AgentState:
    """④ 가드레일 — 근거(도구 결과 + 카테고리 컨텍스트)에 없는 숫자가 있으면 되돌려 보낸다."""
    ok = guardrail(state["answer"], state["results"], state["route"])["ok"]
    return {"guardrail_ok": ok, "action": "ANSWER" if ok else "RETRY"}


def node_escalate(state: AgentState) -> AgentState:
    """어떤 경로로 왔든(확신도 낮음/범위밖/가드레일 반복 위반) action을 종결값으로 정리한다.
    안 그러면 가드레일 위반으로 온 경우 action이 직전 값("RETRY")에 그대로 남아 화면에 이상하게 노출된다."""
    reason = {"ESCALATE": "분류확신도미달", "OUT_OF_SCOPE": "응대범위밖"}.get(
        state["action"], "가드레일위반")
    if state["action"] == "OUT_OF_SCOPE":
        msg = "해당 문의는 이 안내로 답변드리기 어려운 부분입니다. 학사지원팀으로 문의해 주세요."
        final_action = "OUT_OF_SCOPE"
    else:
        msg = escalate_to_agent(reason, {"q": state["question"]})["message"]
        final_action = "ESCALATE"
    return {"answer": msg, "guardrail_ok": state.get("guardrail_ok"), "action": final_action}


def after_route(state: AgentState) -> str:
    return "answer" if state["action"] == "HANDLE" else "escalate"


def after_answer(state: AgentState) -> str:
    return END if state["action"] == "ASK" else "guard"


def after_guard(state: AgentState) -> str:
    """통과하면 끝. 위반이면 한 번 더 생성해 보고, 그래도 안 되면 이관한다."""
    if state["guardrail_ok"]:
        return END
    return "answer" if state.get("attempts", 0) < 1 + GUARDRAIL_RETRY else "escalate"


def build_agent(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("route", node_route)
    g.add_node("answer", node_answer)
    g.add_node("guard", node_guard)
    g.add_node("escalate", node_escalate)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"answer": "answer", "escalate": "escalate"})
    g.add_conditional_edges("answer", after_answer, {"guard": "guard", END: END})
    g.add_conditional_edges("guard", after_guard,
                            {"answer": "answer", "escalate": "escalate", END: END})
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer)


agent_app = build_agent()
chat_app = build_agent(checkpointer=InMemorySaver())     # 대화용(맥락 유지)


def customer_agent(question):
    """문의 한 줄을 파이프라인에 통과시킨다."""
    out = agent_app.invoke({"question": question})
    if out["action"] == "RETRY":
        out["action"] = "ESCALATE"
    return {"question": question, **out}
