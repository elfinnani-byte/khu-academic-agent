# -*- coding: utf-8 -*-
"""조회 도구 — 조문 검색 + 구조화 표 조회 + 이관. 답변은 반드시 이 도구들이 돌려준 값에만 근거해야 한다."""
import json
from pathlib import Path

from context import search_in_category

DATA = Path(__file__).parent / "data"

_scholarship_table = None
_tuition_table = None


def _load_scholarship_table():
    global _scholarship_table
    if _scholarship_table is None:
        _scholarship_table = json.loads((DATA / "scholarship_table.json").read_text(encoding="utf-8"))
    return _scholarship_table


def _load_tuition_table():
    global _tuition_table
    if _tuition_table is None:
        _tuition_table = json.loads((DATA / "tuition_refund_table.json").read_text(encoding="utf-8"))
    return _tuition_table


def search_regulation(category: str, keyword: str) -> dict:
    """카테고리별 조문 조각에서 키워드로 관련 조문을 찾는다."""
    hits = search_in_category(category, keyword)
    if not hits:
        return {"found": False, "articles": []}
    return {"found": True, "articles": [{"source": f, "article": a, "text": t} for f, a, t in hits]}


def get_scholarship_info(name: str) -> dict:
    """별표1 장학금 종류에서 이름으로 조회한다. 모델이 금액·기준을 지어내지 못하게 한다."""
    table = _load_scholarship_table()
    matches = [s for s in table["scholarships"] if name in s["name"] or s["name"] in name]
    if not matches:
        return {"found": False, "scholarships": []}
    return {"found": True, "scholarships": matches}


def get_tuition_refund_rate(days_elapsed) -> dict:
    """학기 개시일로부터 경과일수를 받아 등록금 반환 비율을 조회한다. days_elapsed<0 또는 None이면 개시일 이전."""
    table = _load_tuition_table()
    brackets = table["brackets"]
    if days_elapsed is None or days_elapsed < 0:
        b = brackets[0]
    elif days_elapsed <= 30:
        b = brackets[1]
    elif days_elapsed <= 60:
        b = brackets[2]
    elif days_elapsed <= 90:
        b = brackets[3]
    else:
        b = brackets[4]
    return {"found": True, "days_elapsed": days_elapsed, **b}


def escalate_to_agent(reason: str, payload: dict) -> dict:
    """학사지원팀(사람 상담원)에게 이관한다(목 함수)."""
    return {"message": f"학사지원팀으로 문의해 주시기 바랍니다. (사유: {reason})", "payload": payload}
