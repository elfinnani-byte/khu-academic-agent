# -*- coding: utf-8 -*-
"""Gradio 데모 + 평가 대시보드 + 개선 기록. `python app.py`로 실행 (포트는 config.GRADIO_PORT).

- 💬 상담 데모: 질문 → 답변 + 카테고리/확신도/호출도구/근거/가드레일 통과여부
- 📊 성능 벤치마크: 라우팅 정확도·macro F1·범위밖 인식률·혼동행렬 + 답변 통과율·실패사례
- 📈 개선 기록: 무엇을 바꿔서 수치가 어떻게 변했는지 회차별로 남기고 표로 비교
"""
import json
import re
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

import evaluate as ev
from agent import chat_app
from config import GRADIO_PORT, LABELS4, ROUTE_LABEL_KO, ROUTES
from prompts import ANSWER_RULES, ROUTE_GUIDE

LOG_PATH = Path(__file__).parent / "experiment_log.csv"
LOG_COLUMNS = ["round", "timestamp", "change_target", "change_reason", "change_detail",
               "router_acc", "router_macro_f1", "outscope_recall", "answer_pass_rate", "memo"]

_POLICY_DOC_TEXT = (Path(__file__).parent / "data" / "policy_academic.md").read_text(encoding="utf-8")


def _split_policy_sections(text):
    """policy_academic.md를 '## N. 카테고리 ...' 제목 기준으로 쪼갠다 — (전체 서두, [(제목, 본문), ...])."""
    parts = re.split(r"\n## (\d+\..*)\n", text)
    preamble = parts[0].strip()
    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        body = re.sub(r"\n+---\s*$", "", body).strip()
        sections.append((title, body))
    return preamble, sections


_POLICY_PREAMBLE, _POLICY_SECTIONS = _split_policy_sections(_POLICY_DOC_TEXT)

EVAL_N, OUTSCOPE_N, ANSWER_N = 60, 20, 16  # 각 평가셋의 고정 문항 수(evaluate.py 기준)

ROUTE_LABEL = {
    "ENROLL_REG": "📘 수강·등록",
    "ACADEMIC_STATUS": "🪪 학적",
    "SCHOLARSHIP": "💰 장학·등록금",
    "STUDENT_LIFE": "🏫 생활",
    "OTHER": "❓ 범위밖",
}
ACTION_BADGE = {
    "HANDLE": "🟢 처리",
    "ANSWER": "🟢 답변 완료",
    "ASK": "🟡 재질문",
    "ESCALATE": "🟠 부서 이관",
    "OUT_OF_SCOPE": "🔴 부서 이관",
    "RETRY": "🟡 재생성 중",
}
EXAMPLES = [
    "휴학하려면 어떻게 신청해야 하나요?",
    "성적우수장학금 받으려면 어떤 조건이 필요한가요?",
    "학기 시작하고 15일 지나서 휴학했는데 등록금 얼마나 돌려받나요?",
    "동아리 만들려면 회원이 몇 명 필요한가요?",
    "오늘 학생식당 메뉴가 뭔가요?",
]


# ───────────────────────── 💬 상담 데모 (대화창 + 관제 패널) ─────────────────────────
def user_submit(message, history):
    """1단계: 질문을 대화창에 즉시 올리고 입력창을 비운다(아직 답은 안 만듦)."""
    if not message or not message.strip():
        return history or [], message
    history = (history or []) + [{"role": "user", "content": message.strip()}]
    return history, ""


def bot_respond(history, thread_id):
    """2단계: 대화 맨 끝의 사용자 질문을 실제로 처리해 답변과 관제 정보를 채운다.

    pending을 별도 State로 들고 다니지 않고 history 끝에서 직접 읽는다 — 빠르게 연속 제출될 때
    중간 State가 비어 답변이 조용히 누락되는 문제를 없애기 위함. 이미 답변이 달려 있으면(마지막이
    assistant면) 아무 것도 하지 않는다(중복 호출에 안전).
    """
    history = history or []
    if not history or history[-1]["role"] != "user":
        return history, thread_id, "-", "-", "-", "-", "[]", {}
    pending = history[-1]["content"]
    if not isinstance(pending, str):  # Chatbot content가 리스트/딕트로 올 때 방어
        pending = pending[0] if isinstance(pending, (list, tuple)) and pending else pending
        pending = pending.get("text", "") if isinstance(pending, dict) else str(pending or "")
    if not thread_id:
        thread_id = str(uuid.uuid4())

    out = chat_app.invoke({"question": pending}, config={"configurable": {"thread_id": thread_id}})
    route, conf, action = out.get("route"), out.get("confidence"), out.get("action", "-")
    tools, results, guard_ok = out.get("tools") or [], out.get("results") or {}, out.get("guardrail_ok")

    route_disp = ROUTE_LABEL.get(route, route or "-")
    conf_disp = f"{conf:.2f}" if conf is not None else "-"
    action_disp = ACTION_BADGE.get(action, action)
    guard_disp = "-" if guard_ok is None else ("✅ 정상 (검증 통과)" if guard_ok else "⚠️ 근거 불충분 (재생성됨)")
    tools_disp = json.dumps(tools, ensure_ascii=False) if tools else "[] (호출 없음, 근거 문서만으로 답변)"

    answer = out.get("answer", "(답변 없음)")
    history = (history or []) + [{"role": "assistant", "content": answer}]
    return history, thread_id, route_disp, conf_disp, action_disp, guard_disp, tools_disp, (results or {})


def reset_chat():
    return [], "", str(uuid.uuid4()), "-", "-", "-", "-", "[]", {}


# ───────────────────────── 📊 성능 벤치마크 ─────────────────────────
def _stat_cards(cards):
    """(라벨, 값, 부연설명, 배경색) 튜플들을 카드 형태 HTML로 렌더링한다."""
    items = "".join(
        f'<div style="flex:1;min-width:140px;background:{bg};border-radius:10px;padding:14px;text-align:center;">'
        f'<div style="font-size:0.8em;color:#666;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:1.6em;font-weight:700;color:#222;">{value}</div>'
        f'<div style="font-size:0.75em;color:#888;margin-top:2px;">{sub}</div></div>'
        for label, value, sub, bg in cards
    )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 2px;">{items}</div>'


def _banner(n_bad, n_total, unit="문항", detail_note="아래 표에서 확인하세요."):
    if n_total == 0:
        return ""
    if n_bad == 0:
        return (f'<div style="background:#e8f9ee;border:1px solid #b7e4c7;border-radius:10px;padding:12px;'
                f'text-align:center;font-weight:600;color:#1e5631;">✅ 오류 0건 — {n_total}{unit} 전수 통과했습니다.</div>')
    return (f'<div style="background:#fdeaea;border:1px solid #f3b7b7;border-radius:10px;padding:12px;'
            f'text-align:center;font-weight:600;color:#7a1f1f;">⚠️ {n_bad}건 실패 / {n_total}{unit} — '
            f'{detail_note}</div>')


def _route_disp(code):
    """라우트 코드를 사람이 바로 알아볼 수 있게 'CODE (한글)' 형태로 바꾼다."""
    ko = ROUTE_LABEL_KO.get(code)
    return f"{code} ({ko})" if ko else code


_MISS_COLUMNS = ["question", "gold", "pred", "confidence"]


def _miss_df(miss):
    """오분류 목록(일반 4-카테고리 / 범위밖) 공통 표 생성 — gold·pred를 한글 표기로 바꾼다."""
    if not miss:
        return pd.DataFrame(columns=_MISS_COLUMNS)
    disp = [{**m, "gold": _route_disp(m["gold"]), "pred": _route_disp(m["pred"])} for m in miss]
    return pd.DataFrame(disp, columns=_MISS_COLUMNS)


def _cls_report_df(cls_report, labels):
    rows = []
    for lbl in labels:
        d = cls_report[lbl]
        rows.append([_route_disp(lbl), f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1-score']:.3f}",
                     int(d["support"]), "✅ 만점" if d["f1-score"] >= 0.999 else f"{d['f1-score'] * 100:.0f}%"])
    ma = cls_report["macro avg"]
    rows.append(["단순 평균 (Macro Avg)", f"{ma['precision']:.3f}", f"{ma['recall']:.3f}", f"{ma['f1-score']:.3f}",
                 int(ma["support"]), "-"])
    return pd.DataFrame(rows, columns=["라우트", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "평가 건수", "판정"])


FIRST_COL_PCT = 18  # 1열(라우트 이름) 폭 - 표들끼리 맞추기 위해 %값 자체를 고정한다


def _table_col_widths(df):
    """1열은 표들끼리 폭(%)을 맞추고, 나머지 열은 남은 폭을 균등하게 나눈다.
    전부 %로만 구성해 합이 정확히 100%가 되게 한다 - 고정 px를 하나라도 섞으면
    그만큼 100% 위에 더 얹혀서 가로 스크롤이 생긴다."""
    n_data_cols = len(df.columns) - 1
    if n_data_cols <= 0:
        return None
    share = round((100 - FIRST_COL_PCT) / n_data_cols, 2)
    return [f"{FIRST_COL_PCT}%"] + [f"{share}%"] * n_data_cols


def _cm_cell_html(val, bg, color, weight=600):
    return (f'<div style="background:{bg};color:{color};font-weight:{weight};'
            f'padding:3px 6px;border-radius:4px;text-align:center;">{val}</div>')


def _cm_df(cm_list, row_labels, col_labels):
    """대각선(정답 적중)은 초록, 대각선 밖의 0보다 큰 칸(오분류)은 빨강으로 배경을 칠해서
    표만 훑어봐도 어디서 헷갈렸는지 바로 보이게 한다. 열에는 OTHER(범위밖)까지 포함돼서
    "실제로는 답할 수 있었는데 범위밖으로 잘못 넘긴" 오분류도 빠짐없이 잡힌다.

    gr.Dataframe에 pandas Styler를 넘겨도 이 Gradio 버전에서는 배경색이 실제로 렌더링되지
    않아(직접 확인함), 각 셀 값을 인라인 스타일이 적용된 HTML 문자열로 미리 만들어
    datatype="html" 컬럼에 넣는 방식으로 우회한다.
    """
    disp_rows = [_route_disp(l) for l in row_labels]
    cm = pd.DataFrame(cm_list, index=disp_rows, columns=list(col_labels))  # 열 제목은 코드 그대로(한글 없이)
    cm["정답 합계"] = cm.sum(axis=1)
    total = cm.sum(axis=0)
    total.name = "예측 합계"
    cm = pd.concat([cm, total.to_frame().T])
    cm.insert(0, "실제 정답 \\ 예측", cm.index)
    cm = cm.reset_index(drop=True).astype(object)  # 셀에 HTML 문자열을 넣을 것이므로 숫자 dtype을 풀어둔다

    label_col = "실제 정답 \\ 예측"
    data_cols = [c for c in cm.columns if c != label_col]
    disp_to_code = {_route_disp(l): l for l in row_labels}

    for i in cm.index:
        row_label = cm.loc[i, label_col]  # 한글이 붙은 표시명이거나 "예측 합계"
        is_total_row = row_label == "예측 합계"
        row_code = disp_to_code.get(row_label)  # 대각선 판정은 코드 기준으로 열 이름과 맞춰봐야 한다
        for col in data_cols:
            val = cm.loc[i, col]
            if is_total_row or col == "정답 합계":
                cm.loc[i, col] = _cm_cell_html(val, "#f4f4f7", "#555")
            elif col == row_code:
                cm.loc[i, col] = _cm_cell_html(val, "#d9f5e3", "#1e5631", 700)
            elif val:
                cm.loc[i, col] = _cm_cell_html(val, "#fdeaea", "#7a1f1f")
            else:
                cm.loc[i, col] = _cm_cell_html(val, "transparent", "inherit", 400)
    return cm


# 마지막 측정 결과를 디스크에 남겨서, 서버를 재시작하거나 페이지를 새로고침해도
# (자동 재측정 없이) 마지막으로 실제 측정된 값이 계속 보이도록 한다.
BENCH_CACHE_PATH = Path(__file__).parent / "bench_cache.json"


def _load_bench_cache():
    if BENCH_CACHE_PATH.exists():
        return json.loads(BENCH_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_bench_cache(key, result):
    cache = _load_bench_cache()
    cache[key] = {"measured_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "result": result}
    BENCH_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


SECTION_TITLES = {"router": "① 의도 분류(라우팅) 결과", "answer": "② 1턴 답변 결과"}


def _ts_caption(key):
    """제목과 '마지막 측정' 시각을 한 줄로 합친다 — 시각 부분만 옅은 글씨로 작게 표시."""
    entry = _load_bench_cache().get(key)
    ts = f"🕒 마지막 측정: {entry['measured_at']}" if entry else "아직 측정한 기록이 없습니다."
    note = (f"<span style='font-size:0.75em;font-weight:400;"
            f"color:var(--body-text-color-subdued);margin-left:10px;'>{ts}</span>")
    return f"#### {SECTION_TITLES[key]} {note}"


def _render_router(r):
    n, acc, f1 = r["n"], r["acc"], r["macro_f1"]
    miss = r.get("misclassified", []) or []
    n_miss = len(miss)

    cards = [
        ("평가 데이터셋", f"{n}건", "eval split 전수", "#eef0ff"),
        ("정확도 (Accuracy)", f"{100 * acc:.1f}%", f"{n - n_miss}/{n} 정답", "#e8f9ee" if n_miss == 0 else "#fff7e6"),
        ("Macro F1", f"{f1:.3f}", "4대 카테고리 단순평균", "#eef0ff"),
        ("오분류 건수", f"{n_miss}건", "완벽 적중" if n_miss == 0 else "확인 필요", "#e8f9ee" if n_miss == 0 else "#fdeaea"),
    ]
    if "outscope_recall" in r:
        cards.append(("범위밖 인식률", f"{100 * r['outscope_recall']:.1f}%",
                       "outscope 문항 중 OTHER로 분류", "#eef0ff"))

    stats_html = _stat_cards(cards)
    banner_html = _banner(n_miss, n, unit="건", detail_note="아래 '오분류 목록'에서 확인하세요.")
    cls_df = _cls_report_df(r["classification_report"], LABELS4)
    cm_df = _cm_df(r["confusion_matrix"], LABELS4, r.get("cm_col_labels", ROUTES))
    miss_df = _miss_df(miss)
    outscope_miss = r.get("outscope_misclassified", []) or []
    outscope_miss_df = _miss_df(outscope_miss)
    miss_caption = (f"**오분류 목록 ({n_miss}/{n})** — 4개 카테고리 사이에서 헷갈린 경우"
                     "(정답 라우트가 있는데 다른 라우트로 예측)")
    outscope_caption = (f"**범위밖 오분류 목록 ({len(outscope_miss)}/{OUTSCOPE_N})** — 원래 OTHER(범위밖)로 "
                         "분류돼야 할 질문인데 엉뚱한 카테고리로 예측한 경우(gold는 항상 OTHER). "
                         "범위밖 인식률이 100%가 아닌 이유가 여기 있습니다.")
    return stats_html, banner_html, cls_df, cm_df, miss_df, outscope_miss_df, miss_caption, outscope_caption


def run_router_bench():
    """① 의도 분류(라우팅) 성능만 따로 측정한다. 평가셋 전체를 무조건 다 돈다(샘플링 없음)."""
    r = ev.eval_router(report=True, sample=None)
    _save_bench_cache("router", r)
    return (_ts_caption("router"), *_render_router(r))


def _render_answer(r):
    n, rate = r["n"], r["pass_rate"]
    n_pass = round(rate * n)
    bad = r.get("self_check_bad", [])

    cards = [
        ("평가 데이터셋", f"{n}건", "골든셋 1턴 기준", "#eef0ff"),
        ("통과율 (Pass Rate)", f"{100 * rate:.1f}%", f"{n_pass}/{n} 통과", "#e8f9ee" if n_pass == n else "#fff7e6"),
        ("채점기 자기검증", f"{len(bad)}건 실패" if bad else "이상 없음",
         "모범답안이 채점 기준을 통과하는지", "#fdeaea" if bad else "#e8f9ee"),
    ]
    stats_html = _stat_cards(cards)
    banner_html = _banner(n - n_pass, n, unit="건", detail_note="아래 '실패 사례 자세히 보기'를 펼쳐 확인하세요.")
    fail_df = pd.DataFrame(r.get("failures", []) or [], columns=["conv", "기대", "실제", "fails", "answer"])
    return stats_html, banner_html, fail_df


def run_answer_bench():
    """② 1턴 답변 성능만 따로 측정한다. 골든셋 전체를 무조건 다 돈다(샘플링 없음)."""
    r = ev.eval_answer(report=True, sample=None)
    _save_bench_cache("answer", r)
    return (_ts_caption("answer"), *_render_answer(r))


def _cached_router_outs():
    cached = _load_bench_cache().get("router")
    if not cached:
        empty_cls = pd.DataFrame(columns=["라우트", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "평가 건수", "판정"])
        empty_miss = pd.DataFrame(columns=_MISS_COLUMNS)
        return (_ts_caption("router"), "", "", empty_cls, pd.DataFrame(), empty_miss, empty_miss,
                "**오분류 목록**", "**범위밖 오분류 목록**")
    return (_ts_caption("router"), *_render_router(cached["result"]))


def _cached_answer_outs():
    cached = _load_bench_cache().get("answer")
    if not cached:
        empty_fail = pd.DataFrame(columns=["conv", "기대", "실제", "fails", "answer"])
        return (_ts_caption("answer"), "", "", empty_fail)
    return (_ts_caption("answer"), *_render_answer(cached["result"]))


# ───────────────────────── 📈 개선 기록 ─────────────────────────
LOG_DISPLAY_COLUMNS = ["회차", "변경 대상", "변경 이유(Why)", "변경내용(What)",
                       "의도 분류 정확도", "답변 통과율", "성과 및 오답 메모"]


def _load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _fmt_metric(cur, prev, pct):
    """이전 회차 값과 비교한 증감을 괄호로 같이 보여준다. 첫 회차(prev 없음)는 증감 표시 없이 값만."""
    if pd.isna(cur):
        return "-"
    val = f"{100 * cur:.1f}%" if pct else f"{cur:.3f}"
    if prev is None or pd.isna(prev):
        return val
    diff = cur - prev
    unit = "%p" if pct else ""
    diff_disp = f"{100 * abs(diff):.1f}{unit}" if pct else f"{abs(diff):.3f}"
    if abs(diff) < (0.0005 if pct else 0.0005):
        return f"{val} (=)"
    return f"{val} ({'▲' if diff > 0 else '▼'}{diff_disp})"


def _safe_str(v):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else str(v)


def _fmt_router_cell(acc, f1, prev_acc):
    """정확도(직전 회차 대비 증감) + macro F1 점수를 2줄로 담는다."""
    if pd.isna(acc):
        return "-"
    acc_disp = _fmt_metric(acc, prev_acc, pct=True)
    f1_disp = f"F1 {f1:.3f}" if not pd.isna(f1) else "F1 -"
    return f"{acc_disp}<br>{f1_disp}"


def _fmt_answer_cell(rate, prev_rate):
    """통과율(직전 회차 대비 증감) + 통과 건수를 2줄로 담는다."""
    if pd.isna(rate):
        return "-"
    rate_disp = _fmt_metric(rate, prev_rate, pct=True)
    n_pass = round(rate * ANSWER_N)
    return f"{rate_disp}<br>{n_pass}/{ANSWER_N}건"


def _fmt_memo_cell(acc, outscope_recall, memo, prev_acc, prev_outscope_recall):
    """자유 메모와 자동 계산되는 오분류·범위밖 인식 건수를 합쳐 보여준다.
    직전 회차와 값이 완전히 같은 항목은 굳이 다시 안 적는다(베이스라인은 비교 대상이 없어 둘 다 보여준다).
    메모 안에 {AUTO} 자리표시자를 넣으면 그 위치에 자동 계산 줄이 끼워지고, 없으면 메모 뒤에 이어붙는다
    — 자동 계산 줄을 자유 메모 문장들 사이 원하는 위치에 넣고 싶을 때 쓴다."""
    def _n_miss(a):
        return None if a is None or pd.isna(a) else round((1 - a) * EVAL_N)

    def _n_hit(o):
        return None if o is None or pd.isna(o) else round(o * OUTSCOPE_N)

    parts = []
    n_miss, prev_n_miss = _n_miss(acc), _n_miss(prev_acc)
    if n_miss is not None and n_miss != prev_n_miss:
        parts.append(f"오분류 {n_miss}건")
    n_hit, prev_n_hit = _n_hit(outscope_recall), _n_hit(prev_outscope_recall)
    if n_hit is not None and n_hit != prev_n_hit:
        parts.append(f"범위밖 인식 {n_hit}/{OUTSCOPE_N}건({100 * outscope_recall:.1f}%)")
    auto_block = "<br>".join(f"- {p}" for p in parts)

    memo_text = _safe_str(memo)
    if "{AUTO}" in memo_text:
        merged = memo_text.replace("{AUTO}", auto_block)
        merged = "<br>".join(seg for seg in merged.split("<br>") if seg.strip())
        return merged or "-"
    lines = [memo_text] if memo_text != "-" else []
    if auto_block:
        lines.append(auto_block)
    return "<br>".join(lines) if lines else "-"


def _log_display_df(log):
    """CSV 원본(숫자 그대로)을 사람이 바로 읽을 수 있는 표로 바꾼다 — 컬럼명을 한글로, 지표는 직전 회차 대비 증감을 같이 표시."""
    if log.empty:
        return pd.DataFrame(columns=LOG_DISPLAY_COLUMNS)
    rows = []
    for i in range(len(log)):
        r = log.iloc[i]
        prev = log.iloc[i - 1] if i > 0 else None
        prev_acc = prev["router_acc"] if prev is not None else None
        prev_rate = prev["answer_pass_rate"] if prev is not None else None
        prev_outscope = prev["outscope_recall"] if prev is not None else None
        rows.append({
            "회차": int(r["round"]),
            "변경 대상": _safe_str(r.get("change_target")),
            "변경 이유(Why)": _safe_str(r.get("change_reason")),
            "변경내용(What)": _safe_str(r.get("change_detail")),
            "의도 분류 정확도": _fmt_router_cell(r["router_acc"], r["router_macro_f1"], prev_acc),
            "답변 통과율": _fmt_answer_cell(r["answer_pass_rate"], prev_rate),
            "성과 및 오답 메모": _fmt_memo_cell(r["router_acc"], r["outscope_recall"], r.get("memo"),
                                          prev_acc, prev_outscope),
        })
    return pd.DataFrame(rows, columns=LOG_DISPLAY_COLUMNS)


def refresh_log():
    return _log_display_df(_load_log())


def _run_metrics(sample):
    """개선 기록 한 회차를 남길 때 두 지표를 한 번에 다시 잰다."""
    sample = int(sample) if sample else None
    r_router = ev.eval_router(report=False, sample=sample)
    r_answer = ev.eval_answer(report=False, sample=sample)
    return r_router, r_answer


def add_log_entry(change_target, change_reason, change_detail, memo, sample):
    r_router, r_answer = _run_metrics(sample)
    log = _load_log()
    row = {
        "round": len(log),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "change_target": change_target or "",
        "change_reason": change_reason or "",
        "change_detail": change_detail or "",
        "router_acc": round(r_router["acc"], 3),
        "router_macro_f1": round(r_router["macro_f1"], 3),
        "outscope_recall": round(r_router.get("outscope_recall", float("nan")), 3),
        "answer_pass_rate": round(r_answer["pass_rate"], 3),
        "memo": memo or "",
    }
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    log.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
    status = f"✅ {row['round']}회차로 기록했습니다 (정확도 {row['router_acc']:.3f} · macro F1 {row['router_macro_f1']:.3f})"
    return log, status


THEME = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="slate",
    font=[gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="대학교 학사 안내 에이전트") as demo:
    gr.Markdown(
        "## 🎓 대학교 학사 안내 에이전트\n"
        "수강·등록 · 학적 · 장학·등록금 · 생활 규정을 안내합니다. 근거 문서에 없는 내용은 지어내지 않고 학사지원팀으로 이관합니다."
    )

    with gr.Tabs():
        with gr.Tab("💬 상담 데모"):
            gr.Markdown(
                "### 실시간 상담 데모\n"
                "질문을 입력하면 답변과 함께, 에이전트가 내부적으로 어떤 판단을 했는지(분류·확신도·가드레일 등)를 "
                "오른쪽 패널에서 그대로 확인할 수 있습니다."
            )
            thread_state = gr.State("")
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="상담 대화창", height=430)
                    q = gr.Textbox(label="", placeholder="고객 문의를 입력하세요 (예: 휴학하려면 어떻게 신청하나요?)",
                                    show_label=False)
                    with gr.Row():
                        send_btn = gr.Button("전송", variant="primary")
                        reset_btn = gr.Button("대화 새로 시작")
                    gr.Examples(examples=EXAMPLES, inputs=q, label="추천 질문 바로 해보기")
                with gr.Column(scale=2):
                    gr.Markdown("#### 에이전트 내부 관제")
                    with gr.Row():
                        gr.Markdown("분류 라우트", elem_classes="ins-label")
                        route_out = gr.Textbox(show_label=False, container=False, interactive=False, scale=2)
                    with gr.Row():
                        gr.Markdown("확신도", elem_classes="ins-label")
                        conf_out = gr.Textbox(show_label=False, container=False, interactive=False, scale=2)
                    with gr.Row():
                        gr.Markdown("판정 행동", elem_classes="ins-label")
                        action_out = gr.Textbox(show_label=False, container=False, interactive=False, scale=2)
                    with gr.Row():
                        gr.Markdown("가드레일", elem_classes="ins-label")
                        guard_out = gr.Textbox(show_label=False, container=False, interactive=False, scale=2)
                    with gr.Row():
                        gr.Markdown("호출된 도구", elem_classes="ins-label")
                        tools_out = gr.Textbox(show_label=False, container=False, interactive=False, scale=2)
                    gr.Markdown("도구 조회 결과 원본 (JSON)")
                    evidence_out = gr.JSON(show_label=False)

            submit_outs = [chatbot, q]
            respond_outs = [chatbot, thread_state, route_out, conf_out, action_out, guard_out, tools_out, evidence_out]
            reset_outs = [chatbot, q, thread_state, route_out, conf_out, action_out, guard_out, tools_out, evidence_out]

            send_btn.click(user_submit, inputs=[q, chatbot], outputs=submit_outs) \
                    .then(bot_respond, inputs=[chatbot, thread_state], outputs=respond_outs)
            q.submit(user_submit, inputs=[q, chatbot], outputs=submit_outs) \
             .then(bot_respond, inputs=[chatbot, thread_state], outputs=respond_outs)
            reset_btn.click(reset_chat, outputs=reset_outs)

        with gr.Tab("📊 성능 벤치마크"):
            gr.Markdown(
                "### 핵심 성능 지표 실시간 측정\n"
                "라우팅과 답변, 두 성능을 각각 따로 측정할 수도, 한 번에 같이 측정할 수도 있습니다. "
                "버튼을 누르면 평가셋 전체를 실제로 다시 추론해서 아래 표를 새로 채웁니다(샘플링 없이 항상 전수 평가).\n\n"
                "베이스라인처럼 두 지표를 같은 시점 기준으로 비교하려면 아래 **①+② 전체 재측정**을 누르세요 "
                "— 두 버튼을 각각 누르면 서로 다른 시점에 측정된 값이 섞일 수 있습니다."
            )
            both_btn = gr.Button("①+② 전체 재측정 (라우팅 + 답변 동시)", variant="secondary")
            with gr.Row():
                with gr.Column():
                    router_btn = gr.Button("① 의도 분류만 채점 실행 (eval 60건 + outscope 20건)", variant="primary")
                with gr.Column():
                    answer_btn = gr.Button("② 1턴 답변 통과율만 채점 실행 (골든셋 16건)", variant="primary")

            # value=고정값이 아니라 콜러블을 넘긴다 — Gradio가 "페이지 로드마다" 다시 불러주므로
            # 서버 재시작 없이도 브라우저 새로고침만으로 bench_cache.json의 최신 내용이 반영된다.
            # (실제 채점을 다시 돌리는 게 아니라 이미 저장된 캐시를 다시 읽어 표만 새로 그리는 것이라
            # "페이지 로드시 자동 채점 금지" 원칙과 충돌하지 않는다.)
            _router_init = _cached_router_outs()  # 컬럼 스키마(개수)만 참고 — 값 자체는 아래에서 콜러블로 대체
            router_ts_out = gr.Markdown(lambda: _cached_router_outs()[0], elem_classes="section-heading")
            router_stats_out = gr.HTML(lambda: _cached_router_outs()[1])
            router_banner_out = gr.HTML(lambda: _cached_router_outs()[2])
            with gr.Accordion("라우트별 세부 성능표 · 혼동 행렬 · 오분류 목록 · 범위밖 오분류 목록 자세히 보기", open=False):
                with gr.Group():
                    gr.Markdown("**라우트별 세부 성능 평가표 (Classification Report)**")
                    cls_out = gr.Dataframe(value=lambda: _cached_router_outs()[3], buttons=[], wrap=True,
                                            column_widths=_table_col_widths(_router_init[3]),
                                            elem_classes="fit-table")
                with gr.Group():
                    gr.Markdown(
                        "**혼동 행렬 (Confusion Matrix)** — 행(실제 정답) → 열(모델 예측)\n\n"
                        "🟩 초록 칸(대각선) = 예측이 적중한 건수 · 🟥 빨강 칸 = 오분류된 건수(0보다 큰 칸만 강조)"
                    )
                    cm_out = gr.Dataframe(value=lambda: _cached_router_outs()[4], buttons=[], wrap=True, datatype="html",
                                           column_widths=_table_col_widths(_router_init[4]),
                                           elem_classes="fit-table")
                with gr.Group():
                    miss_caption_out = gr.Markdown(lambda: _cached_router_outs()[7])
                    miss_out = gr.Dataframe(value=lambda: _cached_router_outs()[5], buttons=[], wrap=True,
                                             elem_classes="fit-table",
                                             column_widths=["52%", "16%", "16%", "16%"])
                with gr.Group():
                    outscope_caption_out = gr.Markdown(lambda: _cached_router_outs()[8])
                    outscope_miss_out = gr.Dataframe(value=lambda: _cached_router_outs()[6], buttons=[], wrap=True,
                                                      elem_classes="fit-table",
                                                      column_widths=["52%", "16%", "16%", "16%"])

            _answer_init = _cached_answer_outs()  # 컬럼 스키마만 참고
            answer_ts_out = gr.Markdown(lambda: _cached_answer_outs()[0], elem_classes="section-heading")
            answer_stats_out = gr.HTML(lambda: _cached_answer_outs()[1])
            answer_banner_out = gr.HTML(lambda: _cached_answer_outs()[2])
            with gr.Accordion("실패 사례 자세히 보기", open=False):
                fail_out = gr.Dataframe(value=lambda: _cached_answer_outs()[3], buttons=[], wrap=True,
                                         elem_classes="fit-table")

            router_outs = [router_ts_out, router_stats_out, router_banner_out, cls_out, cm_out, miss_out,
                            outscope_miss_out, miss_caption_out, outscope_caption_out]
            answer_outs = [answer_ts_out, answer_stats_out, answer_banner_out, fail_out]
            router_btn.click(run_router_bench, inputs=None, outputs=router_outs)
            answer_btn.click(run_answer_bench, inputs=None, outputs=answer_outs)
            both_btn.click(run_router_bench, inputs=None, outputs=router_outs) \
                    .then(run_answer_bench, inputs=None, outputs=answer_outs)

        with gr.Tab("📈 개선 기록"):
            gr.Markdown(
                "### 1. 실험 목표\n"
                "① 의도 분류 정확도·macro F1, ② 범위밖 인식률, ③ 1턴 답변 통과율 — 세 지표를 함께 끌어올리되, "
                "하나가 좋아지는 대신 다른 하나가 나빠지는 트레이드오프를 놓치지 않는 것이 목표입니다.\n\n"
                "**실험 원칙**\n"
                "1. 평가셋은 프롬프트에 넣지 않는다 — fewshot 문항만 예시로 쓰고, eval·outscope 문항은 채점 전용으로만 쓴다\n"
                "2. 한 번에 하나만 바꾼다 — 여러 변경을 묶어서 재보면 어느 변경이 효과를 냈는지 알 수 없다\n"
                "3. 바뀐 것과 숫자를 같이 기록한다 — 무엇을·왜 바꿨는지와 세 지표가 어떻게 움직였는지를 한 회차로 남긴다"
            )
            gr.Markdown(
                "### 2. 종합 실험 기록표\n"
                "무엇을 왜 바꿨고 수치가 어떻게 움직였는지 회차별로 남긴 기록입니다. "
                "회차는 성능 벤치마크 탭에서 측정한 뒤 여기에 추가되며, 이 탭 자체에는 입력칸이 없습니다."
            )
            refresh_btn = gr.Button("표 새로고침")
            log_table = gr.Dataframe(value=lambda: _log_display_df(_load_log()), wrap=True, buttons=[], elem_classes="fit-table",
                                      column_widths=["4%", "10%", "20%", "22%", "12%", "10%", "22%"],
                                      datatype=["str", "str", "str", "str", "html", "html", "html"])

            refresh_btn.click(refresh_log, outputs=log_table)

        with gr.Tab("📋 라우트 및 업무 규정"):
            gr.Markdown(
                "### 에이전트가 따르는 분류·응대 기준\n"
                "에이전트가 문의를 어떻게 나누고(라우팅) 어떤 규칙으로 답하는지(응대) 그대로 보여줍니다. "
                "`prompts.py`에 실제로 들어있는 프롬프트 원문이며, 라우팅 결과가 이상해 보일 때 가장 먼저 확인할 곳입니다."
            )
            with gr.Accordion("① 라우팅 분류 기준 (ROUTE_GUIDE)", open=True):
                gr.Markdown(f"```\n{ROUTE_GUIDE.strip()}\n```", elem_classes="wrap-code")
            with gr.Accordion("② 카테고리별 근거 문서 매핑 (policy_academic.md)", open=False):
                gr.Markdown(_POLICY_PREAMBLE)
                for title, body in _POLICY_SECTIONS:
                    with gr.Accordion(title, open=False):
                        gr.Markdown(body)
            with gr.Accordion("③ 답변 생성 규칙 (ANSWER_RULES)", open=False):
                gr.Markdown(f"```\n{ANSWER_RULES.strip()}\n```", elem_classes="wrap-code")

INSPECTOR_CSS = """
.ins-label { min-width: 88px; display: flex; align-items: center;
             font-size: 0.9em; color: var(--body-text-color-subdued); background: none !important; }
.ins-label p { margin: 0; background: none !important; }
.wrap-code pre, .wrap-code pre code {
    white-space: pre-wrap !important;
    word-break: break-word;
    overflow-wrap: anywhere;
    overflow-x: hidden !important;
}
.section-heading, .section-heading > div {
    overflow: visible !important;
}
.section-heading h4 { margin: 0; }
.fit-table table { table-layout: fixed; width: 100% !important; }
.fit-table, .fit-table * { font-size: 0.82rem !important; }
.fit-table td, .fit-table th {
    padding: 4px 6px !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
}
"""

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT, theme=THEME, css=INSPECTOR_CSS)
