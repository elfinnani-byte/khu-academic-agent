# -*- coding: utf-8 -*-
"""Gradio 데모 + 평가 대시보드 + 개선 기록. `python app.py`로 실행 (포트는 config.GRADIO_PORT).

- 💬 상담 데모: 질문 → 답변 + 카테고리/확신도/호출도구/근거/가드레일 통과여부
- 📊 성능 벤치마크: 라우팅 정확도·macro F1·범위밖 인식률·혼동행렬 + 답변 통과율·실패사례
- 📈 개선 기록: 무엇을 바꿔서 수치가 어떻게 변했는지 회차별로 남기고 표로 비교
"""
import json
import uuid
from pathlib import Path

import gradio as gr
import pandas as pd

import evaluate as ev
from agent import chat_app
from config import GRADIO_PORT, LABELS4
from prompts import ANSWER_RULES, ROUTE_GUIDE

LOG_PATH = Path(__file__).parent / "experiment_log.csv"
LOG_COLUMNS = ["round", "timestamp", "change_target", "change_reason", "change_detail",
               "router_acc", "router_macro_f1", "outscope_recall", "answer_pass_rate", "memo"]

_POLICY_DOC_TEXT = (Path(__file__).parent / "data" / "policy_academic.md").read_text(encoding="utf-8")

ROUTE_LABEL = {
    "ENROLL_REG": "📘 수강·등록",
    "ACADEMIC_STATUS": "🎓 학적",
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


def _banner(n_bad, n_total, unit="문항"):
    if n_total == 0:
        return ""
    if n_bad == 0:
        return (f'<div style="background:#e8f9ee;border:1px solid #b7e4c7;border-radius:10px;padding:12px;'
                f'text-align:center;font-weight:600;color:#1e5631;">🎉 오류 0건 — {n_total}{unit} 전수 통과했습니다.</div>')
    return (f'<div style="background:#fdeaea;border:1px solid #f3b7b7;border-radius:10px;padding:12px;'
            f'text-align:center;font-weight:600;color:#7a1f1f;">⚠️ {n_bad}건 실패 / {n_total}{unit} — '
            f'아래 표에서 실패 사례를 확인하세요.</div>')


def _cls_report_df(cls_report, labels):
    rows = []
    for lbl in labels:
        d = cls_report[lbl]
        rows.append([lbl, f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1-score']:.3f}", int(d["support"]),
                     "✅ 만점" if d["f1-score"] >= 0.999 else f"{d['f1-score'] * 100:.0f}%"])
    ma = cls_report["macro avg"]
    rows.append(["단순 평균 (Macro Avg)", f"{ma['precision']:.3f}", f"{ma['recall']:.3f}", f"{ma['f1-score']:.3f}",
                 int(ma["support"]), "-"])
    return pd.DataFrame(rows, columns=["라우트", "정밀도(Precision)", "재현율(Recall)", "F1-Score", "평가 건수", "판정"])


def _cm_df(cm_list, labels):
    cm = pd.DataFrame(cm_list, index=labels, columns=labels)
    cm["정답 합계"] = cm.sum(axis=1)
    total = cm.sum(axis=0)
    total.name = "예측 합계"
    cm = pd.concat([cm, total.to_frame().T])
    cm.insert(0, "실제 정답 \\ 예측", cm.index)
    return cm.reset_index(drop=True)


def run_router_bench():
    """① 의도 분류(라우팅) 성능만 따로 측정한다. 평가셋 전체를 무조건 다 돈다(샘플링 없음)."""
    r = ev.eval_router(report=True, sample=None)
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
    banner_html = _banner(n_miss, n, unit="건")
    cls_df = _cls_report_df(r["classification_report"], LABELS4)
    cm_df = _cm_df(r["confusion_matrix"], LABELS4)
    miss_df = pd.DataFrame(miss, columns=["question", "gold", "pred", "confidence"]) if miss else \
        pd.DataFrame(columns=["question", "gold", "pred", "confidence"])
    return stats_html, banner_html, cls_df, cm_df, miss_df


def run_answer_bench():
    """② 1턴 답변 성능만 따로 측정한다. 골든셋 전체를 무조건 다 돈다(샘플링 없음)."""
    r = ev.eval_answer(report=True, sample=None)
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
    banner_html = _banner(n - n_pass, n, unit="건")
    fail_df = pd.DataFrame(r.get("failures", []) or [], columns=["conv", "기대", "실제", "fails", "answer"])
    return stats_html, banner_html, fail_df


# ───────────────────────── 📈 개선 기록 ─────────────────────────
LOG_DISPLAY_COLUMNS = ["회차", "변경 대상", "바뀐 이유", "바뀐 내용",
                       "정확도", "Macro F1", "범위밖 인식률", "답변 통과율", "결과 메모"]


def _load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def _fmt_metric(cur, prev, pct):
    """이전 회차 값과 비교한 증감을 괄호로 같이 보여준다. 첫 회차(prev 없음)는 '(기준)'."""
    if pd.isna(cur):
        return "-"
    val = f"{100 * cur:.1f}%" if pct else f"{cur:.3f}"
    if prev is None or pd.isna(prev):
        return f"{val} (기준)"
    diff = cur - prev
    unit = "%p" if pct else ""
    diff_disp = f"{100 * abs(diff):.1f}{unit}" if pct else f"{abs(diff):.3f}"
    if abs(diff) < (0.0005 if pct else 0.0005):
        return f"{val} (=)"
    return f"{val} ({'▲' if diff > 0 else '▼'}{diff_disp})"


def _safe_str(v):
    return "-" if v is None or (isinstance(v, float) and pd.isna(v)) or v == "" else str(v)


def _log_display_df(log):
    """CSV 원본(숫자 그대로)을 사람이 바로 읽을 수 있는 표로 바꾼다 — 컬럼명을 한글로, 지표는 직전 회차 대비 증감을 같이 표시."""
    if log.empty:
        return pd.DataFrame(columns=LOG_DISPLAY_COLUMNS)
    rows = []
    for i in range(len(log)):
        r = log.iloc[i]
        prev = log.iloc[i - 1] if i > 0 else None
        rows.append({
            "회차": int(r["round"]),
            "변경 대상": _safe_str(r.get("change_target")),
            "바뀐 이유": _safe_str(r.get("change_reason")),
            "바뀐 내용": _safe_str(r.get("change_detail")),
            "정확도": _fmt_metric(r["router_acc"], prev["router_acc"] if prev is not None else None, pct=True),
            "Macro F1": _fmt_metric(r["router_macro_f1"], prev["router_macro_f1"] if prev is not None else None, pct=False),
            "범위밖 인식률": _fmt_metric(r["outscope_recall"], prev["outscope_recall"] if prev is not None else None, pct=True),
            "답변 통과율": _fmt_metric(r["answer_pass_rate"], prev["answer_pass_rate"] if prev is not None else None, pct=True),
            "결과 메모": _safe_str(r.get("memo")),
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


THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")

with gr.Blocks(title="대학교 학사 안내 에이전트") as demo:
    gr.Markdown(
        "## 🎓 대학교 학사 안내 에이전트\n"
        "수강·등록 · 학적 · 장학·등록금 · 생활 규정을 안내합니다. 근거 문서에 없는 내용은 지어내지 않고 학사지원팀으로 이관합니다."
    )

    with gr.Tabs():
        with gr.Tab("💬 상담 데모"):
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
                    gr.Markdown("**에이전트 내부 관제**")
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
                    gr.Markdown("**① 의도 분류(라우팅) 성능** — eval 60건 + outscope 20건 전체")
                    router_btn = gr.Button("의도 분류만 채점 실행", variant="primary")
                with gr.Column():
                    gr.Markdown("**② 1턴 답변 성능** — 골든셋 16건 전체")
                    answer_btn = gr.Button("1턴 답변 통과율만 채점 실행", variant="primary")

            gr.Markdown("#### ① 의도 분류(라우팅) 결과")
            router_stats_out = gr.HTML()
            router_banner_out = gr.HTML()
            with gr.Accordion("라우트별 세부 성능표 · 혼동 행렬 · 오분류 목록 자세히 보기", open=False):
                gr.Markdown("**라우트별 세부 성능 평가표 (Classification Report)**")
                cls_out = gr.Dataframe()
                gr.Markdown("**혼동 행렬 (Confusion Matrix)** — 대각선이 아닌 칸은 오분류된 건수")
                cm_out = gr.Dataframe()
                gr.Markdown("**오분류 목록**")
                miss_out = gr.Dataframe()

            gr.Markdown("#### ② 1턴 답변 결과")
            answer_stats_out = gr.HTML()
            answer_banner_out = gr.HTML()
            with gr.Accordion("실패 사례 자세히 보기", open=False):
                fail_out = gr.Dataframe()

            router_outs = [router_stats_out, router_banner_out, cls_out, cm_out, miss_out]
            answer_outs = [answer_stats_out, answer_banner_out, fail_out]
            router_btn.click(run_router_bench, inputs=None, outputs=router_outs)
            answer_btn.click(run_answer_bench, inputs=None, outputs=answer_outs)
            both_btn.click(run_router_bench, inputs=None, outputs=router_outs) \
                    .then(run_answer_bench, inputs=None, outputs=answer_outs)

        with gr.Tab("📈 개선 기록"):
            gr.Markdown(
                "### 종합 실험 기록표\n"
                "무엇을 왜 바꿨고 수치가 어떻게 움직였는지 회차별로 남긴 기록입니다. "
                "원칙: 평가셋은 프롬프트에 넣지 않기 · 한 번에 하나만 바꾸기 · 바뀐 것과 숫자를 같이 기록하기.\n\n"
                "회차는 성능 벤치마크 탭에서 측정한 뒤 여기에 추가되며, 이 탭 자체에는 입력칸이 없습니다."
            )
            refresh_btn = gr.Button("표 새로고침")
            log_table = gr.Dataframe(value=_log_display_df(_load_log()), wrap=True)

            refresh_btn.click(refresh_log, outputs=log_table)

        with gr.Tab("📋 라우트 및 업무 규정"):
            gr.Markdown(
                "### 에이전트가 따르는 분류·응대 기준\n"
                "에이전트가 문의를 어떻게 나누고(라우팅) 어떤 규칙으로 답하는지(응대) 그대로 보여줍니다. "
                "`prompts.py`에 실제로 들어있는 프롬프트 원문이며, 라우팅 결과가 이상해 보일 때 가장 먼저 확인할 곳입니다."
            )
            with gr.Accordion("① 라우팅 분류 기준 (ROUTE_GUIDE)", open=True):
                gr.Markdown(f"```\n{ROUTE_GUIDE.strip()}\n```")
            with gr.Accordion("② 답변 생성 규칙 (ANSWER_RULES)", open=False):
                gr.Markdown(f"```\n{ANSWER_RULES.strip()}\n```")
            with gr.Accordion("③ 카테고리별 근거 문서 매핑 (policy_academic.md)", open=False):
                gr.Markdown(_POLICY_DOC_TEXT)

INSPECTOR_CSS = """
.ins-label { min-width: 88px; display: flex; align-items: center;
             font-size: 0.9em; color: var(--body-text-color-subdued); background: none !important; }
.ins-label p { margin: 0; background: none !important; }
"""

if __name__ == "__main__":
    demo.launch(server_port=GRADIO_PORT, theme=THEME, css=INSPECTOR_CSS)
