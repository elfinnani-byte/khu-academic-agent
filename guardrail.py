# -*- coding: utf-8 -*-
"""가드레일 — 답변 속 숫자의 출처를 근거(도구 결과 + 카테고리 컨텍스트)에서 역추적한다.

모두몰과 다른 점: 우리는 카테고리 조문 전체를 프롬프트에 직접 주입하므로(context.build_context),
도구를 안 부르고도 정답을 낼 수 있다. 그래서 근거 풀을 "도구 결과"뿐 아니라
"카테고리 컨텍스트 텍스트"까지 합쳐서 검사해야 정상적인 답변을 오탐하지 않는다.
"""
import json
import re

from context import build_context

_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _norm(s) -> str:
    """1,000 같은 표기의 쉼표를 지워 숫자만 남긴다."""
    return re.sub(r"(?<=\d),(?=\d)", "", str(s))


def guardrail(answer: str, results: dict, route: str) -> dict:
    """답변에 등장하는 숫자가 근거(도구 결과 + 카테고리 컨텍스트) 어딘가에 실제로 있는지 확인한다."""
    pool = _norm(json.dumps(results, ensure_ascii=False)) + "\n" + _norm(build_context(route))
    nums = {_norm(m.group()) for m in _NUM_RE.finditer(answer)}
    ungrounded = sorted(n for n in nums if n and n not in pool)
    return {"ok": not ungrounded, "ungrounded_numbers": ungrounded}
