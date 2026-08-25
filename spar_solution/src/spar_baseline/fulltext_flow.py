"""正文驱动的检索流程（对齐用户 2026-08-25 定稿的新流程）。

流程：种子论文 → 获取正文（arXiv HTML 优先，PDF 本地抽取兜底）→ 切章节
→ LLM 按固定指令挑相关章节 → 从引用句解析被引论文 → 按标题查回真论文
→ 双评审互评 → 交卷。

设计约束（用户裁定）：
- 不读正文不得推荐；
- Sub-agent（扩展员）每篇种子一个，产出结构化、带理由；
- 评审 = 双 Agent 互评（两个不同立场的固定指令），不搞五人圆桌；
- 所有 LLM 调用有固定 prompt 模板 + JSON 输出 + 降级路径。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen

from .fulltext import _fetch, fulltext_text, query_terms


ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

SECTION_SELECTOR_PROMPT = (
    "You are a careful research assistant selecting paper sections to expand from. "
    "Given a research question and a paper's sections (title + preview), pick ONLY the sections "
    "whose content could cite work relevant to answering the question. "
    "Return JSON only: {\"relevant\": [section indices], \"reason\": \"<one short sentence>\"}. "
    "Skip references/appendix/acknowledgments sections."
)

CITATION_PICKER_PROMPT = (
    "You are a citation expansion specialist. Given a research question and citation "
    "contexts extracted from relevant sections of a paper (each with the citing sentence "
    "and the referenced work), pick the 2-4 referenced works MOST likely to be relevant "
    "answers to the question. Return JSON only: "
    "{\"picks\": [{\"ref_index\": <n>, \"reason\": \"<short>\"}]}."
)

REFS_PICKER_PROMPT = (
    "You are a citation expansion specialist. Given a research question and the "
    "reference list (verbatim, may be numbered or author-year style) of a relevant paper, "
    "pick the 2-4 references MOST likely to be relevant answers to the question. "
    "Return JSON only: {\"picks\": [{\"query\": \"<the reference's paper title, cleaned for search>\", \"reason\": \"<short>\"}]}. "
    "Output the actual paper titles, cleaned of page numbers/venues/arXiv ids."
)

REVIEWER_A_PROMPT = (
    "You are a strict relevance reviewer for academic search. For each candidate paper "
    "(title+abstract) versus the research question, return JSON only: "
    "{\"verdicts\": [{\"paper_n\": <n>, \"score\": 0.0-1.0, \"reason\": \"<short>\"}]}. "
    "Score >=0.7 only if the paper directly answers the question; surveys/tangential work <=0.4."
)

REVIEWER_B_PROMPT = (
    "You are a pragmatic recall-oriented reviewer: a relevant paper may use different "
    "terminology than the question. For each candidate versus the question, return JSON "
    "only: {\"verdicts\": [{\"paper_n\": <n>, \"score\": 0.0-1.0, \"reason\": \"<short>\"}]}. "
    "Score >=0.6 when the method or object of study plausibly matches, even if wording differs."
)


class FulltextFlowError(RuntimeError):
    pass


def _get(url: str, *, timeout: float = 30.0, max_bytes: int = 6 * 1024 * 1024) -> bytes:
    request = Request(url, headers={"User-Agent": "spar-fulltext-flow/1.0"})
    with urlopen(request, timeout=timeout) as response:  # nosec B310: 学术公开页
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise FulltextFlowError("content exceeds size cap")
    return data


def _strip_tags(fragment: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", fragment, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_arxiv_html_markdown(arxiv_id: str, *, cache_dir: str | Path | None = None) -> str | None:
    """拉取 arXiv 原生 HTML（或 ar5iv 镜像）并转成带编号引用的纯文本。

    cache_dir 提供时先读盘缓存（html/<id>.txt），命中不再联网——正文获取
    是 hybrid 模式的时延大头（实测单题 2.5-6 分钟超红线的主要原因）。
    """

    normalized = re.sub(r"^(arxiv:)?", "", str(arxiv_id or "").strip(), flags=re.IGNORECASE)
    normalized = re.sub(r"v\d+$", "", normalized)
    if not re.fullmatch(r"\d{4}\.\d{4,5}(\.\d+)?|[a-z-]+/\d{7}", normalized):
        return None
    cache_path = None
    if cache_dir is not None:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized)
        cache_path = Path(cache_dir) / "html" / f"{safe}.txt"
        if cache_path.is_file():
            cached = cache_path.read_text(encoding="utf-8", errors="ignore")
            if cached.strip():
                return cached
    for template in (ARXIV_HTML_URL, AR5IV_HTML_URL):
        try:
            html = _get(template.format(arxiv_id=normalized)).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if "References" not in html and "references" not in html:
            continue
        # 保留正文中的 [n] 引用标记：arXiv HTML 的 <a ...>[n]</a> 去标签后自然留下 [n]。
        text = _strip_tags(html)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
        return text
    return None


@dataclass
class PaperFulltext:
    paper_id: str
    source: str  # arxiv_html / pdf / none
    sections: list[dict[str, Any]] = field(default_factory=list)
    references: list[dict[str, Any]] = field(default_factory=list)  # {"index": n, "raw": ...}
    citation_contexts: list[dict[str, Any]] = field(default_factory=list)  # {"ref": n, "sentence": ...}
    references_text: str = ""  # References 节原文（编号/作者-年份格式皆可）


_REF_LINE_RE = re.compile(r"\[(\d{1,3})\]\s+(.{10,400}?)(?=\[\d{1,3}\]|$)")
_CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})+)\]")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def parse_references_and_sections(text: str, *, section_chars: int = 1800) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    """从全文纯文本解析编号参考文献列表、章节与引用句。

    返回 (references, sections, citation_contexts)。
    """

    references: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    if not text:
        return references, sections
    marker = None
    for candidate in ("References", "REFERENCES", "references"):
        position = text.rfind(candidate)
        if position != -1:
            marker = (candidate, position)
            break
    body = text
    refs_text_out = ""
    if marker:
        name, position = marker
        body = text[:position]
        refs_text = text[position:]
        refs_text_out = refs_text[:12000]
        for index, entry in _REF_LINE_RE.findall(refs_text[:20000]):
            references.append({"index": int(index), "raw": entry.strip()[:300]})
    # 章节切分：常见编号标题（1 Introduction / 2.1 xxx / Abstract）。
    parts = re.split(r"(?=\b(?:Abstract|[IVX]+\.?\s+\w|\d{1,2}(?:\.\d)?\s+[A-Z]\w+))", body)
    cursor = 0
    for part in parts:
        if not part.strip():
            continue
        title = part.strip()[:80]
        sections.append({"title": title, "text": part.strip(), "start": cursor})
        cursor += len(part)
    if not sections:
        sections = [{"title": "body", "text": body[: section_chars * 10], "start": 0}]
    for section in sections:
        section["preview"] = section["text"][:220]
        section.pop("text", None)
    citation_contexts: list[dict[str, Any]] = []
    for match in _CITE_RE.finditer(body):
        for number in re.findall(r"\d{1,3}", match.group(1)):
            start = body.rfind(".", 0, match.start()) + 1
            end = body.find(".", match.end())
            sentence = body[max(start, match.start() - 200):end if end != -1 else match.end() + 200].strip()
            citation_contexts.append({"ref": int(number), "sentence": sentence[:320]})
    return references, sections, citation_contexts, refs_text_out


def load_paper_fulltext(paper: Mapping[str, Any], *, cache_dir: str | Path | None = None) -> PaperFulltext:
    """获取单篇论文正文并解析章节/引用。

    优先 arXiv HTML（快、引用带编号），但老论文的 HTML 转换常缺参考文献节
    （实测 1312.7452：无 References、引用为 author-year 键）——解析不出
    参考文献时回落 PDF（PDF 参考文献永远完整），取引用更丰富的一份。
    """

    paper_id = str(paper.get("paper_id") or "")
    arxiv_id = (paper.get("identifiers") or {}).get("arxiv_id")
    best: PaperFulltext | None = None
    if arxiv_id:
        html_text = fetch_arxiv_html_markdown(arxiv_id, cache_dir=cache_dir)
        if html_text:
            references, sections, contexts, refs_text = parse_references_and_sections(html_text)
            best = PaperFulltext(paper_id, "arxiv_html", sections, references, contexts, refs_text)
            if references:
                return best
    pdf_url = (paper.get("access") or {}).get("pdf_url") or (paper.get("access") or {}).get("oa_url")
    if pdf_url and cache_dir is not None:
        from .fulltext import download_pdf

        path = download_pdf(pdf_url, Path(cache_dir) / "pdf")
        if path is not None:
            pdf_text = fulltext_text(path)
            if len(pdf_text) > 200:
                references, sections, contexts, refs_text = parse_references_and_sections(pdf_text)
                pdf_fulltext = PaperFulltext(paper_id, "pdf", sections, references, contexts, refs_text)
                if best is None or len(references) > len(best.references):
                    return pdf_fulltext
    return best if best is not None else PaperFulltext(paper_id, "none")


def extract_reference_title(raw: str) -> str:
    """从参考文献原始文本粗提标题（取较长且像标题的片段）。"""

    fragments = re.split(r",\s+", raw)
    fragments = [fragment.strip(" .\"'") for fragment in fragments if fragment.strip(" .\"'")]
    if not fragments:
        return raw[:120]
    return max(fragments, key=len)[:160]


def select_relevant_sections(client: Any, question: str, fulltext: PaperFulltext, *, max_sections: int = 4) -> list[int]:
    """固定指令：让 LLM 挑与问题相关的章节索引。失败退全部章节（前 max_sections 个）。"""

    if not client or not fulltext.sections:
        return list(range(min(max_sections, len(fulltext.sections))))
    payload = {"question": question, "sections": [{"i": i, "title": s["title"], "preview": s.get("preview", "")[:200]} for i, s in enumerate(fulltext.sections)]}
    try:
        result = client.complete_json(
            SECTION_SELECTOR_PROMPT,
            json.dumps(payload, ensure_ascii=False),
            max_tokens=300,
        )
        indices = [int(i) for i in (result.get("relevant") or []) if isinstance(i, (int, float)) or str(i).isdigit()]
        indices = [i for i in indices if 0 <= i < len(fulltext.sections)][:max_sections]
        return indices or list(range(min(max_sections, len(fulltext.sections))))
    except Exception:
        return list(range(min(max_sections, len(fulltext.sections))))


def pick_citations(client: Any, question: str, fulltext: PaperFulltext, section_indices: Sequence[int], *, max_picks: int = 4) -> list[dict[str, Any]]:
    """从相关章节的引用句中挑最值得扩展的参考文献。失败退词法覆盖率 top-N。"""

    known_refs = {ref["index"] for ref in fulltext.references}
    contexts = [c for c in fulltext.citation_contexts if c["ref"] in known_refs or not known_refs]
    if not contexts:
        return []
    if client is not None:
        items = [{"n": c["ref"], "context": c["sentence"]} for c in contexts[:40]]
        try:
            result = client.complete_json(
                CITATION_PICKER_PROMPT,
                json.dumps({"question": question, "citation_contexts": items}, ensure_ascii=False),
                max_tokens=500,
            )
            picks = []
            for item in result.get("picks") or []:
                try:
                    picks.append({"ref_index": int(item.get("ref_index")), "reason": str(item.get("reason") or "")[:120]})
                except (TypeError, ValueError):
                    continue
            picks = [p for p in picks if any(c["ref"] == p["ref_index"] for c in contexts)][:max_picks]
            if picks:
                return picks
        except Exception:
            pass
    terms = query_terms(question)
    scored = sorted(contexts, key=lambda c: -sum(1 for t in terms if t in c["sentence"].casefold()))
    return [{"ref_index": c["ref"], "reason": "lexical_fallback"} for c in scored[:max_picks]]


def pick_references(client: Any, question: str, fulltext: PaperFulltext, *, max_picks: int = 4) -> list[dict[str, Any]]:
    """让 LLM 直接从参考文献列表点名最相关条目（编号/作者-年份格式通吃）。

    失败兜底：用编号解析条目的词法覆盖度 top-N；再不行取 References 前几行。
    返回 [{"query": 检索用标题, "reason": ...}]。
    """

    refs_text = fulltext.references_text
    if not refs_text and fulltext.references:
        refs_text = "\n".join(str(r.get("raw") or "") for r in fulltext.references)[:12000]
    if not refs_text:
        return []
    if client is not None:
        try:
            result = client.complete_json(
                REFS_PICKER_PROMPT,
                json.dumps({"question": question, "references": refs_text[:10000]}, ensure_ascii=False),
                max_tokens=700,
            )
            picks = []
            for item in result.get("picks") or []:
                query = str(item.get("query") or "").strip()
                if len(query) >= 8:
                    picks.append({"query": query[:160], "reason": str(item.get("reason") or "")[:120]})
            if picks:
                return picks[:max_picks]
        except Exception:
            pass
    terms = query_terms(question)
    scored = sorted(fulltext.references, key=lambda r: -sum(1 for t in terms if t in str(r.get("raw", "")).casefold()))
    return [{"query": extract_reference_title(r.get("raw", "")), "reason": "lexical_fallback"} for r in scored[:max_picks]]


def dual_review(client: Any, question: str, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """双评审互评：严格派 + 召回派。两派都 >= 阈值才入选；分歧进 uncertain。

    返回每篇 {paper_id, score_a, score_b, final, status: keep/uncertain/reject}。
    """

    items = [{"paper_n": i, "title": (p.get("bibliography") or {}).get("title"), "abstract": str((p.get("bibliography") or {}).get("abstract") or "")[:500]} for i, p in enumerate(candidates)]
    scores: dict[int, dict[str, float]] = {i: {"score_a": None, "score_b": None} for i in range(len(items))}
    # 分批评审（每批 5 篇）防响应截断；失败显式记录而不是伪装 0 分。
    chunks = [items[i : i + 5] for i in range(0, len(items), 5)] or [[]]
    for label, prompt in (("score_a", REVIEWER_A_PROMPT), ("score_b", REVIEWER_B_PROMPT)):
        for chunk in chunks:
            if client is None:
                break
            try:
                result = client.complete_json(prompt, json.dumps({"question": question, "candidates": chunk}, ensure_ascii=False), max_tokens=900)
                for verdict in result.get("verdicts") or []:
                    try:
                        n = int(verdict.get("paper_n"))
                        scores[n][label] = max(0.0, min(1.0, float(verdict.get("score") or 0.0)))
                    except (TypeError, ValueError, KeyError):
                        continue
            except Exception:
                for item in chunk:
                    scores[item["paper_n"]][label] = None  # 评审不可用：显式缺失
    output = []
    for i, paper in enumerate(candidates):
        a, b = scores[i]["score_a"], scores[i]["score_b"]
        if a is None and b is None:
            status, final = "review_failed", None
        elif a is None or b is None:
            known = a if a is not None else b
            status = "keep" if known >= 0.6 else "reject"
            final = round(known, 4)
        elif a >= 0.6 and b >= 0.5:
            status, final = "keep", round((a + b) / 2, 4)
        elif a >= 0.35 and b >= 0.35:
            status, final = "uncertain", round((a + b) / 2, 4)
        else:
            status, final = "reject", round((a + b) / 2, 4)
        output.append({"paper_id": str(paper.get("paper_id")), "score_a": a, "score_b": b, "final": final, "status": status})
    return output


__all__ = [
    "PaperFulltext",
    "dual_review",
    "extract_reference_title",
    "fetch_arxiv_html_markdown",
    "load_paper_fulltext",
    "parse_references_and_sections",
    "pick_citations",
    "pick_references",
    "select_relevant_sections",
]
