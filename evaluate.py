# -*- coding: utf-8 -*-
"""두 지표(+범위밖 인식률)를 한 번에 잰다.  `python evaluate.py`

    ① 의도 분류 정확도 - eval 60건, 정답 라우트와 대조 (macro F1은 LABELS4 4개만)
    ①-보조 범위밖 인식률 - outscope 20건, 예측 라우트가 OTHER인 비율
       (모두몰 evaluate.py에는 이 채점이 빠져 있어 우리가 추가함 - AT05-01의
       "넘기기 경로도 재야 한다" 요구사항 충족용)
    ② 1턴 답변 통과율 - 정답셋 16개 대화, action·tools·must·forbid 네 축

`--only router` / `--only answer` 로 한쪽만 잴 수 있고, `--sample N`으로 일부만 재서
API 비용을 아낄 수 있다.
"""
import argparse
import json
import re

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from common import pmap
from config import BASE, LABELS4, WORKERS


# --- ① 의도 분류 ---
def eval_router(report=True, sample=None):
    from router import app

    inq = pd.read_csv(BASE / "student_inquiries.csv")
    ans = pd.read_csv(BASE / "routing_answers.csv")
    gold = inq.merge(ans, on="qa_id")
    ev = gold[gold["split"] == "eval"].reset_index(drop=True)
    outscope = gold[gold["split"] == "outscope"].reset_index(drop=True)
    if sample:
        ev = ev.sample(min(sample, len(ev)), random_state=0).reset_index(drop=True)
        outscope = outscope.sample(min(sample, len(outscope)), random_state=0).reset_index(drop=True)

    states = pmap(lambda q: app.invoke({"question": q}), ev["question"].tolist(), workers=WORKERS)
    pred = [s["route"] for s in states]
    y = ev["route"].tolist()

    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, labels=LABELS4, average="macro", zero_division=0)
    print(f"--- ① 의도 분류 (eval n={len(ev)}) ---")
    print(f"정확도 {acc:.3f}  macro F1 {f1:.3f}")

    result = {"acc": acc, "macro_f1": f1, "n": len(ev)}
    if report:
        cls_report = classification_report(y, pred, labels=LABELS4, digits=3,
                                            zero_division=0, output_dict=True)
        cm = confusion_matrix(y, pred, labels=LABELS4)
        print()
        print(classification_report(y, pred, labels=LABELS4, digits=3, zero_division=0))
        print("[혼동 행렬] 행=정답, 열=예측")
        print(pd.DataFrame(cm, index=LABELS4, columns=LABELS4).to_string())

        miss = [(q, g, p, s["confidence"]) for q, g, p, s in
                zip(ev["question"], y, pred, states) if g != p]
        print(f"\n[오분류 {len(miss)}건] - 여기를 읽는 것이 개선의 출발점이다")
        for q, g, p, conf in miss:
            print(f"  [{g} → {p}] conf={conf:.2f}  {q[:56]}")

        result.update({
            "labels": LABELS4, "confusion_matrix": cm.tolist(),
            "classification_report": cls_report,
            "misclassified": [{"question": q, "gold": g, "pred": p, "confidence": s["confidence"]}
                              for q, g, p, s in zip(ev["question"], y, pred, states) if g != p],
        })

    # ①-보조: 범위밖 인식률 - 모두몰에 없던 지표
    if len(outscope):
        os_states = pmap(lambda q: app.invoke({"question": q}), outscope["question"].tolist(), workers=WORKERS)
        os_pred = [s["route"] for s in os_states]
        os_recall = sum(p == "OTHER" for p in os_pred) / len(os_pred)
        print(f"\n[범위밖 인식률] outscope n={len(outscope)}  OTHER로 정확히 분류 {os_recall:.3f}")
        result["outscope_recall"] = os_recall
        if report:
            os_miss = [q for q, p in zip(outscope["question"], os_pred) if p != "OTHER"]
            if os_miss:
                print(f"[범위밖 오분류 {len(os_miss)}건]")
                for q in os_miss:
                    print(f"  {q[:60]}")
            result["outscope_misclassified"] = os_miss

    return result


# --- ② 1턴 답변 ---
AUTO_ACTIONS = {"ANSWER", "ASK"}   # OUT_OF_SCOPE/ESCALATE는 라우터 단계 결정이라 여기선 안 잼
_ASK_RE = re.compile(r"\?|주시겠|알려주|말씀해")


def norm_num(s):
    return re.sub(r"(?<=\d),(?=\d)", "", str(s))


def score_turn(expect, answer, tools_called, action):
    """한 턴을 채점한다. 반환: (통과 여부, 실패 항목)"""
    fails = []
    a = norm_num(answer)
    if expect["action"] != action:
        fails.append(f'action: 기대 {expect["action"]} != 실제 {action}')
    need = set(expect.get("tools", []))
    if need - set(tools_called):
        fails.append(f'tools 미호출: {sorted(need - set(tools_called))}')
    for m in expect.get("must", []):
        if norm_num(m) not in a:
            fails.append(f'must 누락: "{m}"')
    for f in expect.get("forbid", []):
        if norm_num(f) in a:
            fails.append(f'forbid 위반: "{f}"')
    if expect["action"] == "ASK" and not _ASK_RE.search(answer):
        fails.append("ASK인데 되묻는 문장이 아님")
    return (not fails), fails


def load_cases():
    gold = json.loads((BASE / "answer_goldenset_multiturn.json").read_text(encoding="utf-8"))
    cases = []
    for c in gold["conversations"]:
        q = next(t for t in c["turns"] if t["role"] == "customer")
        a = next((t for t in c["turns"] if t.get("expect")), None)
        if a:
            cases.append({"conv_id": c["conv_id"], "question": q["text"],
                          "route": c["route"], "expect": a["expect"]})
    return cases


def eval_answer(report=True, sample=None):
    from answer import answer_with_tools

    cases = load_cases()
    if sample:
        cases = cases[:sample]

    # 채점기 자체 검증 - 모범 답안은 전부 통과해야 한다. 아니면 채점기가 틀린 것이다.
    bad = [c["conv_id"] for c in cases
           if not score_turn(c["expect"], c["expect"]["reference"],
                             c["expect"].get("tools", []), c["expect"]["action"])[0]]
    print("--- ② 1턴 답변 ---")
    print(f"[채점기 자기 검증] 모범 답안 {len(cases)}건 중 실패 {len(bad)}건 {bad if bad else 'OK'}")

    def run_case(case):
        text, results = answer_with_tools(case["question"], case["route"])
        if not results and _ASK_RE.search(text):
            action = "ASK"
        else:
            action = "ANSWER"
        return action, text, list(results)

    scored = [c for c in cases if c["expect"]["action"] in AUTO_ACTIONS]
    outs = pmap(run_case, scored, workers=WORKERS)

    rows = []
    for c, (action, text, tools_called) in zip(scored, outs):
        ok, fails = score_turn(c["expect"], text, tools_called, action)
        rows.append({"conv": c["conv_id"], "기대": c["expect"]["action"], "실제": action,
                     "ok": ok, "fails": "; ".join(fails), "answer": text})
    res = pd.DataFrame(rows)
    rate = res["ok"].mean()
    print(f'채점 {len(res)}건 / 통과 {res["ok"].sum()}건 ({100 * rate:.1f}%)')

    result = {"pass_rate": rate, "n": len(res), "self_check_bad": bad}
    if report:
        print("\n[행동 판정 혼동] 행=기대, 열=실제")
        print(pd.crosstab(res["기대"], res["실제"]).to_string())
        kinds = [f.split(":")[0] for s in res.loc[~res["ok"], "fails"] for f in s.split("; ") if f]
        print("\n[실패 유형]")
        print(pd.Series(kinds).value_counts().to_string() if kinds else "  없음")
        print("\n[실패 사례] - 여기를 읽는 것이 개선의 출발점이다")
        for _, r in res[~res["ok"]].iterrows():
            print(f'  {r["conv"]} 기대={r["기대"]} 실제={r["실제"]}  {r["fails"][:80]}')
            print(f'      답변: {r["answer"][:90]}')
        result.update({
            "failure_kinds": pd.Series(kinds).value_counts().to_dict() if kinds else {},
            "failures": res.loc[~res["ok"], ["conv", "기대", "실제", "fails", "answer"]].to_dict("records"),
        })

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="두 지표(+범위밖 인식률)를 잰다")
    ap.add_argument("--only", choices=["router", "answer"], help="한쪽만 재기")
    ap.add_argument("--sample", type=int, help="일부만 재서 비용 절약(개발 중 반복 확인용)")
    ap.add_argument("--quiet", action="store_true", help="요약만")
    args = ap.parse_args()

    out = {}
    if args.only != "answer":
        out["router"] = eval_router(report=not args.quiet, sample=args.sample)
        print()
    if args.only != "router":
        out["answer"] = eval_answer(report=not args.quiet, sample=args.sample)

    print("\n=== 요약 ===")
    if "router" in out:
        print(f'  ① 의도 분류   정확도 {out["router"]["acc"]:.3f} · macro F1 {out["router"]["macro_f1"]:.3f}'
              + (f' · 범위밖 인식률 {out["router"]["outscope_recall"]:.3f}' if "outscope_recall" in out["router"] else ""))
    if "answer" in out:
        print(f'  ② 1턴 답변    통과율 {100 * out["answer"]["pass_rate"]:.1f}% ({out["answer"]["n"]}건)')
