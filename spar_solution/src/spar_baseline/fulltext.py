"""全文证据第 1 层：本地下载文本型 PDF 并抽取正文（无需云端 OCR）。

arXiv 上的现代论文绝大多数是文本型 PDF，用 pymupdf 直接抽文字即可，不
需要 PaddleOCR（扫描版留给 tools/ 转换工具的第 2 层）。协议约束：
- 正文永远落盘为 artifact 文件，PaperDoc 只保存 chunk 引用；
- ``full_text_status`` 只有真实抽到正文才升为 ``fulltext``；
- 抽不出文字（扫描版）不算失败，显式返回 skipped；
- 任何下载/解析失败都返回 None/跳过，绝不让论文变低分。
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from .paperdoc import validate_paper_doc


DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_BYTES = 15 * 1024 * 1024
CHUNK_CHARS = 1500
MAX_CHUNKS = 8
# 全文关键词覆盖率对相关性分的小幅修正（精排信号，不是重新打分）。
NUDGE_BONUS = 0.05
NUDGE_PENALTY = 0.05
NUDGE_HIGH = 0.8
NUDGE_LOW = 0.3

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_TERM_RE = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with",
    "about", "papers", "paper", "studies", "study", "using", "based", "are",
    "is", "what", "which", "how", "can", "tell", "me", "some", "any", "there",
}


def _safe_name(value: str) -> str:
    return _SAFE_NAME.sub("_", str(value)).strip("._")[:120] or "paper"


def _fetch(url: str, *, timeout: float, max_bytes: int, opener: Callable[[str, float], bytes] | None = None) -> bytes:
    """下载限制在 max_bytes 内；opener 可注入便于离线测试。"""

    def default(open_url: str, open_timeout: float) -> bytes:
        request = Request(open_url, headers={"User-Agent": "spar-fulltext/1.0"})
        with urlopen(request, timeout=open_timeout) as response:  # nosec B310: 论文 OA PDF 链接
            return response.read(max_bytes + 1)

    data = (opener or default)(url, timeout)
    if len(data) > max_bytes:
        raise ValueError("pdf exceeds size cap")
    return data


def download_pdf(url: str, dest_dir: str | Path, *, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES, opener: Callable[[str, float], bytes] | None = None) -> Path | None:
    """下载 pdf_url 到 dest_dir；失败返回 None。"""

    if not isinstance(url, str) or not url.strip().lower().startswith(("http://", "https://")):
        return None
    try:
        data = _fetch(url.strip(), timeout=timeout, max_bytes=max_bytes, opener=opener)
    except Exception:
        return None
    if not data.startswith(b"%PDF"):
        return None
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (_safe_name(url.rstrip("/").rsplit("/", 1)[-1]) + ".pdf")
    path.write_bytes(data)
    return path


def extract_fulltext_chunks(pdf_path: str | Path, *, chunk_chars: int = CHUNK_CHARS, max_chunks: int = MAX_CHUNKS) -> list[dict[str, Any]] | None:
    """抽取正文并切块；抽不出文字（扫描版）返回 None。"""

    try:
        import fitz  # pymupdf；懒加载，未安装不影响系统其余部分
    except ImportError:
        return None
    try:
        document = fitz.open(str(pdf_path))
    except Exception:
        return None
    try:
        pages: list[str] = []
        for page in document:
            text = page.get_text("text") or ""
            if text.strip():
                pages.append(text)
        full = "\n".join(pages).strip()
        if len(full) < 200:  # 文本太少，大概率是扫描版
            return None
        chunks: list[dict[str, Any]] = []
        offset = 0
        while offset < len(full) and len(chunks) < max_chunks:
            piece = full[offset : offset + chunk_chars]
            chunks.append({
                "chunk_id": f"{Path(pdf_path).stem}:chunk:{len(chunks)}",
                "offset": offset,
                "section": "body",
                "page": None,
                "char_count": len(piece),
            })
            offset += chunk_chars
        return chunks if chunks else None
    finally:
        document.close()


def fulltext_text(pdf_path: str | Path) -> str:
    """读取整段正文文本（落盘与覆盖率计算用）。"""

    try:
        import fitz
    except ImportError:
        return ""
    try:
        document = fitz.open(str(pdf_path))
    except Exception:
        return ""
    try:
        return "\n".join((page.get_text("text") or "") for page in document).strip()
    finally:
        document.close()


def query_terms(query: str | Sequence[str] | None) -> list[str]:
    """从查询/主题提取关键词（去停用词）。"""

    if query is None:
        return []
    text = " ".join(query) if isinstance(query, Sequence) and not isinstance(query, str) else str(query)
    return [term.casefold() for term in _TERM_RE.findall(text) if term.casefold() not in _STOPWORDS and len(term) > 1]


def fulltext_coverage(text: str, terms: Sequence[str]) -> float | None:
    """查询关键词在全文中的覆盖率；无关键词或无正文返回 None。"""

    terms = {term for term in terms}
    if not terms or not text:
        return None
    blob = text.casefold()
    return sum(1 for term in terms if term in blob) / len(terms)


def apply_fulltext_nudge(paper: dict[str, Any], coverage: float | None, *, bonus: float = NUDGE_BONUS, penalty: float = NUDGE_PENALTY, high: float = NUDGE_HIGH, low: float = NUDGE_LOW) -> dict[str, Any]:
    """按全文覆盖率小幅修正相关性分（±0.05），并记录 provenance。"""

    scores = paper.setdefault("scores", {})
    relevance = scores.get("relevance")
    nudge = 0.0
    if coverage is not None and isinstance(relevance, (int, float)):
        if coverage >= high:
            nudge = bonus
        elif coverage < low:
            nudge = -penalty
        if nudge:
            scores["relevance"] = round(max(0.0, min(1.0, float(relevance) + nudge)), 6)
    paper.setdefault("provenance", {})["fulltext"] = {"coverage": coverage, "nudge": nudge}
    return paper


def augment_fulltext(
    paper: Mapping[str, Any],
    cache_dir: str | Path,
    *,
    terms: Sequence[str] | None = None,
    opener: Callable[[str, float], bytes] | None = None,
) -> dict[str, Any] | None:
    """给单篇论文下载 PDF、抽取正文并升级证据状态。

    返回 {"paper": 更新后的 PaperDoc, "text_ref": 正文文件, "coverage": 覆盖率}
    ；无 pdf_url、下载失败或扫描版均返回 None（调用方保持原状即可）。
    """

    doc = dict(paper)
    try:
        validate_paper_doc(doc)
    except Exception:
        return None
    pdf_url = (doc.get("access") or {}).get("pdf_url") or (doc.get("access") or {}).get("oa_url")
    if not pdf_url:
        return None
    root = Path(cache_dir)
    pdf_path = download_pdf(pdf_url, root / "pdf", opener=opener)
    if pdf_path is None:
        return None
    text = fulltext_text(pdf_path)
    chunks = extract_fulltext_chunks(pdf_path)
    if not text or chunks is None:
        return None
    safe_id = _safe_name(str(doc.get("paper_id") or "paper"))
    text_path = root / "fulltext" / f"{safe_id}.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    rel_ref = str(text_path.relative_to(root)).replace("\\", "/")
    for chunk in chunks:
        chunk["content_ref"] = rel_ref
    doc.setdefault("content", {})["chunks"] = chunks
    doc.setdefault("content", {})["char_count"] = len(text)
    doc.setdefault("access", {})["full_text_status"] = "fulltext"
    doc.setdefault("status", {})["evidence_status"] = "fulltext"
    coverage = fulltext_coverage(text, terms) if terms else None
    apply_fulltext_nudge(doc, coverage)
    validate_paper_doc(doc)
    return {"paper": doc, "text_ref": rel_ref, "coverage": coverage, "char_count": len(text)}


def augment_topk(
    papers: list[dict[str, Any]],
    cache_dir: str | Path,
    terms: Sequence[str] | None,
    *,
    top_k: int = 10,
    opener: Callable[[str, float], bytes] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对前 top_k 篇尝试全文增强（其余不动），返回 (papers, stats)。"""

    stats = {"attempted": 0, "succeeded": 0, "skipped_scanned_or_failed": 0, "top_k": top_k}
    output: list[dict[str, Any]] = []
    for index, paper in enumerate(papers):
        if index < top_k and ((paper.get("access") or {}).get("pdf_url") or (paper.get("access") or {}).get("oa_url")):
            stats["attempted"] += 1
            result = augment_fulltext(paper, cache_dir, terms=terms, opener=opener)
            if result is None:
                stats["skipped_scanned_or_failed"] += 1
            else:
                stats["succeeded"] += 1
                paper = result["paper"]
        output.append(paper)
    return output, stats


__all__ = [
    "augment_fulltext",
    "augment_topk",
    "apply_fulltext_nudge",
    "download_pdf",
    "extract_fulltext_chunks",
    "fulltext_coverage",
    "fulltext_text",
    "query_terms",
]
