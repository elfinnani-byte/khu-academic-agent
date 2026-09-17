# -*- coding: utf-8 -*-
"""여러 모듈이 함께 쓰는 잡동사니. 어제 실습(모두몰)의 common.py를 그대로 이식."""
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

# 프로바이더마다 예외 클래스 이름이 다르다(openai.RateLimitError 등). 클래스를 일일이
# 임포트하는 대신 이름·메시지에 이 표시어가 있으면 요청 한도로 본다.
RATE_LIMIT_MARKERS = ("ratelimiterror", "rate_limit", "resource_exhausted", "429", "quota")

# "하루 몇 건/몇 토큰" 류 한도라는 표시어. 안내 시간이 없으면 이 한도는 재시도해도 헛수고다.
PER_DAY_MARKERS = ("perday", "requestsperday", "dailylimit", "daily quota", "per day", "tpd")


def _is_rate_limit_error(e):
    text = f"{type(e).__name__} {e}".lower()
    return any(m in text for m in RATE_LIMIT_MARKERS)


def _hinted_wait_seconds(e):
    """'retry in 5.2s', 'try again in 2m46.752s' 같은 안내를 초 단위로 읽는다."""
    m = re.search(r"(?:retry|try again)\D{0,10}(?:(\d+)\s*m)?\D{0,5}(\d+(?:\.\d+)?)\s*s",
                 str(e), re.I)
    if not m:
        return None
    minutes = float(m.group(1)) if m.group(1) else 0
    return minutes * 60 + float(m.group(2))


def _retry_delay(e, attempt, base_delay, max_delay):
    """안내된 대기 시간이 있으면 그 값을(최대 5분까지), 없으면 지수 백오프를 쓴다."""
    hinted = _hinted_wait_seconds(e)
    if hinted is not None:
        base = hinted + 1
        cap = max(max_delay, 300)
    else:
        base = base_delay * (2 ** attempt)
        cap = max_delay
    return min(base + random.uniform(0, base * 0.5), cap)


def call_with_retry(fn, *args, max_retries=5, base_delay=5, max_delay=60, **kwargs):
    """호출 한 건을 실행한다. 요청 한도(429 류)에 걸리면 죽지 않고 기다렸다가 재시도한다."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries or not _is_rate_limit_error(e):
                raise
            text = f"{type(e).__name__} {e}".lower()
            hinted = _hinted_wait_seconds(e)
            if any(m in text for m in PER_DAY_MARKERS) and hinted is None:
                raise
            delay = _retry_delay(e, attempt, base_delay, max_delay)
            print(f"[rate-limit] {type(e).__name__} — {delay:.0f}초 대기 후 재시도 "
                  f"({attempt + 1}/{max_retries})", flush=True)
            time.sleep(delay)


def pmap(fn, items, workers=8, max_retries=5, base_delay=5):
    """여러 건을 동시에 호출한다. 결과 순서는 입력 순서와 같다."""
    def wrapped(item):
        return call_with_retry(fn, item, max_retries=max_retries, base_delay=base_delay)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(wrapped, items))


def show_graph(compiled):
    """그래프 구조를 mermaid 로 출력한다."""
    print(compiled.get_graph().draw_mermaid())
