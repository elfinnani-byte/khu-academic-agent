# -*- coding: utf-8 -*-
"""guide_docs/*.md 원문을 조문 단위로 쪼개고, 카테고리별로 어떤 조문을 프롬프트에 넣을지 조립한다.

policy_academic.md에 적어 둔 카테고리 ↔ 조문 매핑을 그대로 코드로 옮긴 것이 CATEGORY_ARTICLES다.
근거는 요약(policy_academic.md)이 아니라 이 파일이 읽어오는 guide_docs 원문 그대로 쓴다.
"""
import re
from pathlib import Path

GUIDE_DOCS = Path(__file__).parent / "guide_docs"

# 파일명 그대로("KHU_학사운영에_관한_규정.md") 노출하면 모델이 그걸 인용해버리므로,
# 답변에 실제로 노출될 수 있는 모든 지점(build_context, search_in_category)에서
# 사람이 읽는 규정명으로 바꿔서 내보낸다.
DOC_DISPLAY_NAME = {
    "KHU_학칙.md": "학칙",
    "KHU_학사운영에_관한_규정.md": "학사운영에 관한 규정",
    "KHU_장학규정.md": "장학규정",
    "KHU_교내장학금_종류_및_지급기준.md": "교내장학금 종류 및 지급기준(별표1)",
    "KHU_학생생활규정.md": "학생생활규정",
    "KHU_학생상벌에관한규정.md": "학생상벌에관한규정",
}


def display_name(filename: str) -> str:
    return DOC_DISPLAY_NAME.get(filename, filename)

# "### 제25조(...)" 또는 "### 제25조의2(...)" 형태의 조문 제목을 찾는다.
_ARTICLE_RE = re.compile(r"^###\s*제(\d+)조(?:의(\d+))?\(([^)]*)\)")

# 카테고리 ↔ (파일, 조문번호 목록) 매핑. "__ALL__"이면 그 파일 전체를 쓴다(별표1처럼 조문 구조가 없는 표).
CATEGORY_ARTICLES = {
    "ENROLL_REG": {
        "KHU_학사운영에_관한_규정.md": [
            "2", "3", "4", "5", "6", "20", "21", "24", "25", "25의2", "26", "27",
            "28", "29", "30", "31", "32", "33", "34", "34의2", "35", "36", "37",
            "38", "39", "40", "42", "43", "44", "45", "46", "47", "48",
        ],
        "KHU_학칙.md": [
            "6", "7", "8", "9", "10", "11",
            "34", "34의2", "34의3", "34의4", "35", "37", "38", "39", "40", "41",
            "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52",
            "53", "54", "55", "56", "57", "58", "58의2", "59", "60",
        ],
    },
    "ACADEMIC_STATUS": {
        "KHU_학사운영에_관한_규정.md": [
            "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
            "17", "18", "19", "22", "23",
        ],
        "KHU_학칙.md": [
            "23", "24", "25", "26", "27",
            "28", "29", "30", "30의2", "31", "32", "33",
            "36",
        ],
    },
    "SCHOLARSHIP": {
        "KHU_장학규정.md": [str(i) for i in range(1, 17)],
        "KHU_교내장학금_종류_및_지급기준.md": ["__ALL__"],
    },
    "STUDENT_LIFE": {
        "KHU_학생생활규정.md": [str(i) for i in range(1, 21)],
        "KHU_학생상벌에관한규정.md": [str(i) for i in range(1, 22)],
    },
}

_article_cache = {}


def parse_articles(filename):
    """마크다운 파일을 조문 단위로 쪼갠다. {"25의2": {"title":..., "text":...}, ...} 형태로 반환."""
    text = (GUIDE_DOCS / filename).read_text(encoding="utf-8")
    articles = {}
    current_id = None
    current_title = None
    buf = []
    for line in text.split("\n"):
        m = _ARTICLE_RE.match(line)
        if m:
            if current_id is not None:
                articles[current_id] = {"title": current_title, "text": "\n".join(buf).strip()}
            main, sub, title = m.groups()
            current_id = f"{main}의{sub}" if sub else main
            current_title = title
            buf = [line]
        elif current_id is not None:
            buf.append(line)
    if current_id is not None:
        articles[current_id] = {"title": current_title, "text": "\n".join(buf).strip()}
    return articles


def _get_articles(filename):
    if filename not in _article_cache:
        _article_cache[filename] = parse_articles(filename)
    return _article_cache[filename]


def _whole_file_text(filename):
    return (GUIDE_DOCS / filename).read_text(encoding="utf-8")


def build_context(category: str) -> str:
    """카테고리에 해당하는 조문만 모아 하나의 문자열로 조립한다(프롬프트에 넣을 근거)."""
    parts = []
    for filename, article_ids in CATEGORY_ARTICLES.get(category, {}).items():
        if article_ids == ["__ALL__"]:
            parts.append(f"## 출처: {display_name(filename)}\n\n{_whole_file_text(filename)}")
            continue
        arts = _get_articles(filename)
        chunk = [arts[aid]["text"] for aid in article_ids if aid in arts]
        if chunk:
            parts.append(f"## 출처: {display_name(filename)}\n\n" + "\n\n".join(chunk))
    return "\n\n---\n\n".join(parts)


def search_in_category(category: str, keyword: str):
    """카테고리 안에서 키워드가 포함된 조문만 검색한다. [(파일, 조문번호, 본문), ...] 반환."""
    results = []
    for filename, article_ids in CATEGORY_ARTICLES.get(category, {}).items():
        if article_ids == ["__ALL__"]:
            text = _whole_file_text(filename)
            if keyword in text:
                results.append((display_name(filename), "전체", text))
            continue
        arts = _get_articles(filename)
        for aid in article_ids:
            a = arts.get(aid)
            if a and keyword in a["text"]:
                results.append((display_name(filename), aid, a["text"]))
    return results
