"""P3 五 Agent 的结构化消息协议与 artifact 存储。

Agent 之间只交换 JSON artifact 引用和短状态字段。论文、摘要、完整计划等
较大对象不会嵌入消息正文，从协议层避免自然语言长文本复制。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


PROTOCOL_SCHEMA = "p3_message.v1"
PROTOCOL = "spar-agent.v1"
AGENT_ROLES = {"planner", "retriever", "citation_explorer", "evidence_judge", "arbiter"}
PEERS = AGENT_ROLES | {"orchestrator", "system"}
MESSAGE_TYPES = {
    "QUERY_PLAN", "SEARCH_ACTION", "RESULT_BATCH", "RELATION_BATCH",
    "EVIDENCE_VERDICT", "STOP_DECISION", "FINAL_SELECTION",
}
DIAGNOSTIC_CODES = {
    "OK", "DEGRADED", "RATE_LIMIT", "SCHEMA_ERROR", "NO_EVIDENCE",
    "CONFIG_MISSING", "PROVIDER_ERROR", "CONFLICT",
}
_MESSAGE_SENDERS = {
    "QUERY_PLAN": {"planner"},
    "SEARCH_ACTION": {"planner", "orchestrator", "arbiter"},
    "RESULT_BATCH": {"retriever"},
    "RELATION_BATCH": {"citation_explorer"},
    "EVIDENCE_VERDICT": {"evidence_judge"},
    "STOP_DECISION": {"arbiter"},
    "FINAL_SELECTION": {"arbiter"},
}
_RELATION_TYPES = {"references", "citations", "related_works"}
_VERDICT_TYPES = {"relevant", "irrelevant", "uncertain", "degraded"}
_STOP_ACTIONS = {"STOP", "NEXT_QUERY", "FINAL_SELECTION"}
_LONG_KEYS = {"abstract", "body", "content", "document", "full_text", "fulltext", "markdown", "raw_text"}
_REF_RE = re.compile(r"^(?:artifacts/)?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:json|jsonl|md|txt)$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class P3ProtocolError(ValueError):
    """P3 消息、payload 或 artifact 引用不满足协议。"""
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_name(value: str) -> str:
    return _SAFE.sub("_", str(value)).strip("._")[:100] or "artifact"


def _validate_short(value: Any, path: str = "short_fields") -> None:
    if isinstance(value, str):
        if len(value) > 256:
            raise ValueError(f"{path} string exceeds 256 characters")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_short(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        if len(value) > 32:
            raise ValueError(f"{path} list exceeds 32 items")
        for index, item in enumerate(value):
            _validate_short(item, f"{path}[{index}]")
        return
    if value is not None and not isinstance(value, (bool, int, float)):
        raise ValueError(f"{path} contains unsupported value")


@dataclass(frozen=True)
class AgentMessage:
    message_id: str
    run_id: str
    sender: str
    receiver: str
    kind: str
    artifact_ref: str
    short_fields: dict[str, Any]
    schema_version: str = PROTOCOL_SCHEMA

    def __post_init__(self) -> None:
        if self.sender not in AGENT_ROLES or self.receiver not in AGENT_ROLES:
            raise ValueError("sender and receiver must be registered P3 roles")
        if not self.artifact_ref or not isinstance(self.artifact_ref, str):
            raise ValueError("artifact_ref is required")
        if "body" in self.short_fields or "text" in self.short_fields:
            raise ValueError("long natural-language body is not allowed")
        _validate_short(self.short_fields)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        run_id: str,
        sender: str,
        receiver: str,
        kind: str,
        artifact_ref: str,
        short_fields: Mapping[str, Any] | None = None,
    ) -> "AgentMessage":
        seed = f"{run_id}|{sender}|{receiver}|{kind}|{artifact_ref}"
        message_id = "msg_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return cls(message_id, run_id, sender, receiver, kind, artifact_ref, dict(short_fields or {}))


class ArtifactStore:
    """将阶段对象保存为 JSON，并返回相对于运行根目录的引用。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, stage: str, payload: Any, *, name: str | None = None) -> str:
        filename = _safe_name(name or stage) + ".json"
        path = self.root / _safe_name(stage) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return str(path.relative_to(self.root)).replace("\\", "/")

    def read(self, ref: str) -> Any:
        path = (self.root / ref).resolve()
        root = self.root.resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(ref)
        return json.loads(path.read_text(encoding="utf-8"))

    def write_jsonl(self, name: str, rows: list[Mapping[str, Any]]) -> str:
        path = self.root / _safe_name(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(_json(row) + "\n" for row in rows), encoding="utf-8")
        return str(path.relative_to(self.root)).replace("\\", "/")


def estimate_bytes(value: Any) -> int:
    return len(_json(value).encode("utf-8"))


def estimate_tokens(value: Any) -> int:
    return max(1, (estimate_bytes(value) + 3) // 4)


def estimate_message(message: AgentMessage) -> dict[str, int]:
    size = estimate_bytes(message.to_dict())
    return {"bytes": size, "tokens_estimate": max(1, (size + 3) // 4)}


def _require_string(value: Any, path: str, *, max_length: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise P3ProtocolError(f"{path} must be a non-empty string")
    if len(value) > max_length:
        raise P3ProtocolError(f"{path} exceeds inline string limit {max_length}")
    return value


def _require_id(value: Any, path: str) -> str:
    text = _require_string(value, path, max_length=128)
    if not _ID_RE.fullmatch(text):
        raise P3ProtocolError(f"{path} contains unsupported identifier characters")
    return text


def _require_list(value: Any, path: str, *, allow_empty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise P3ProtocolError(f"{path} must be an array")
    return value


def _require_number(value: Any, path: str, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise P3ProtocolError(f"{path} must be a number")
    result = float(value)
    if result < minimum or result > maximum:
        raise P3ProtocolError(f"{path} is outside the allowed range")
    return result


def _check_inline(value: Any, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise P3ProtocolError(f"{path} keys must be strings")
            if key.casefold() in _LONG_KEYS:
                raise P3ProtocolError(f"{path}.{key} must be stored in an artifact")
            _check_inline(item, f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) > 128:
            raise P3ProtocolError(f"{path} list exceeds 128 items")
        for index, item in enumerate(value):
            _check_inline(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) > 512:
            raise P3ProtocolError(f"{path} exceeds inline string limit 512")
    elif value is not None and not isinstance(value, (bool, int, float)):
        raise P3ProtocolError(f"{path} contains a non-JSON value")


def validate_artifact_ref(value: Any, *, field: str = "artifact_ref") -> str:
    """校验相对 artifact 引用，拒绝绝对路径、URL 和路径穿越。"""

    ref = _require_string(value, field)
    if "\\" in ref or ".." in Path(ref).parts or not _REF_RE.fullmatch(ref):
        raise P3ProtocolError(f"{field} must be a safe relative JSON/JSONL/MD/TXT reference")
    return ref


def resolve_artifact_ref(ref: str, root: str | Path) -> Path:
    validate_artifact_ref(ref)
    base = Path(root).resolve()
    candidate = (base / ref).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise P3ProtocolError("artifact reference escapes root") from exc
    return candidate


def _string_ids(value: Any, path: str) -> list[str]:
    return [_require_id(item, f"{path}[{index}]") for index, item in enumerate(_require_list(value, path))]


def _validate_payload(kind: str, payload: Mapping[str, Any]) -> None:
    _check_inline(payload)
    if kind == "QUERY_PLAN":
        _require_id(payload.get("query_id"), "payload.query_id")
        rows = _require_list(payload.get("subqueries"), "payload.subqueries", allow_empty=False)
        seen: set[str] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise P3ProtocolError(f"payload.subqueries[{index}] must be an object")
            sid = _require_id(row.get("subquery_id"), f"payload.subqueries[{index}].subquery_id")
            if sid in seen:
                raise P3ProtocolError("duplicate subquery_id")
            seen.add(sid)
            kind_name = _require_string(row.get("kind"), f"payload.subqueries[{index}].kind", max_length=32)
            if kind_name not in {"topic", "method", "dataset", "constraint", "comparison", "reference"}:
                raise P3ProtocolError(f"unsupported subquery kind: {kind_name}")
            _require_string(row.get("query_text"), f"payload.subqueries[{index}].query_text")
            _string_ids(row.get("source_capabilities"), f"payload.subqueries[{index}].source_capabilities")
    elif kind == "SEARCH_ACTION":
        for field in ("action_id", "query_id", "subquery_id", "source"):
            _require_id(payload.get(field), f"payload.{field}")
        if not isinstance(payload.get("page"), int) or isinstance(payload.get("page"), bool) or payload["page"] < 0:
            raise P3ProtocolError("payload.page must be a non-negative integer")
        if not isinstance(payload.get("page_size"), int) or isinstance(payload.get("page_size"), bool) or not 1 <= payload["page_size"] <= 100:
            raise P3ProtocolError("payload.page_size must be an integer between 1 and 100")
    elif kind == "RESULT_BATCH":
        for field in ("batch_id", "query_id", "source"):
            _require_id(payload.get(field), f"payload.{field}")
        _string_ids(payload.get("paper_ids"), "payload.paper_ids")
        for field in ("records_ref", "provenance_ref"):
            if payload.get(field) is not None:
                validate_artifact_ref(payload[field], field=f"payload.{field}")
    elif kind == "RELATION_BATCH":
        for field in ("relation_batch_id", "query_id", "source"):
            _require_id(payload.get(field), f"payload.{field}")
        for index, row in enumerate(_require_list(payload.get("edges"), "payload.edges")):
            if not isinstance(row, Mapping):
                raise P3ProtocolError(f"payload.edges[{index}] must be an object")
            _require_id(row.get("parent_paper_id"), f"payload.edges[{index}].parent_paper_id")
            _require_id(row.get("child_paper_id"), f"payload.edges[{index}].child_paper_id")
            relation = _require_string(row.get("relation_type"), f"payload.edges[{index}].relation_type", max_length=32)
            if relation not in _RELATION_TYPES:
                raise P3ProtocolError(f"unsupported relation_type: {relation}")
            depth = row.get("depth")
            if not isinstance(depth, int) or isinstance(depth, bool) or depth not in (0, 1):
                raise P3ProtocolError("relation depth must be 0 or 1")
    elif kind == "EVIDENCE_VERDICT":
        _require_id(payload.get("query_id"), "payload.query_id")
        _require_id(payload.get("paper_id"), "payload.paper_id")
        verdict = _require_string(payload.get("verdict"), "payload.verdict", max_length=32)
        if verdict not in _VERDICT_TYPES:
            raise P3ProtocolError(f"unsupported verdict: {verdict}")
        state = _require_string(payload.get("constraint_state"), "payload.constraint_state", max_length=16)
        if state not in {"pass", "fail", "unknown"}:
            raise P3ProtocolError("invalid constraint_state")
        _require_number(payload.get("confidence"), "payload.confidence")
        refs = _require_list(payload.get("evidence_refs"), "payload.evidence_refs")
        for index, ref in enumerate(refs):
            validate_artifact_ref(ref, field=f"payload.evidence_refs[{index}]")
        scores = payload.get("component_scores")
        if not isinstance(scores, Mapping):
            raise P3ProtocolError("payload.component_scores must be an object")
        for field in ("relevance", "constraint", "evidence", "quality", "citation", "novelty"):
            _require_number(scores.get(field), f"payload.component_scores.{field}")
    elif kind == "STOP_DECISION":
        _require_id(payload.get("query_id"), "payload.query_id")
        action = _require_string(payload.get("action"), "payload.action", max_length=32)
        if action not in _STOP_ACTIONS:
            raise P3ProtocolError(f"unsupported stop action: {action}")
        _require_string(payload.get("reason_code"), "payload.reason_code", max_length=64)
    elif kind == "FINAL_SELECTION":
        _require_id(payload.get("query_id"), "payload.query_id")
        for index, row in enumerate(_require_list(payload.get("selections"), "payload.selections")):
            if not isinstance(row, Mapping):
                raise P3ProtocolError(f"payload.selections[{index}] must be an object")
            _require_id(row.get("paper_id"), f"payload.selections[{index}].paper_id")
            _require_number(row.get("final_score"), f"payload.selections[{index}].final_score")
            for ref_index, ref in enumerate(_require_list(row.get("evidence_refs"), f"payload.selections[{index}].evidence_refs")):
                validate_artifact_ref(ref, field=f"payload.selections[{index}].evidence_refs[{ref_index}]")


def validate_payload(message_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if message_type not in MESSAGE_TYPES:
        raise P3ProtocolError(f"unsupported message type: {message_type}")
    if not isinstance(payload, Mapping):
        raise P3ProtocolError("payload must be an object")
    _validate_payload(message_type, payload)
    return json.loads(json.dumps(payload, ensure_ascii=False))


def validate_message(message: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        raise P3ProtocolError("message must be an object")
    required = {"protocol", "run_id", "message_id", "type", "sender", "receiver", "seq", "payload"}
    missing = required - set(message)
    if missing:
        raise P3ProtocolError(f"message missing fields: {sorted(missing)}")
    if message["protocol"] not in {PROTOCOL, PROTOCOL_SCHEMA}:
        raise P3ProtocolError("unsupported protocol")
    for field in ("run_id", "message_id", "sender", "receiver"):
        _require_id(message[field], f"message.{field}")
    message_type = _require_string(message["type"], "message.type", max_length=32)
    if message_type not in MESSAGE_TYPES:
        raise P3ProtocolError(f"unsupported message type: {message_type}")
    if message["sender"] not in PEERS or message["receiver"] not in PEERS:
        raise P3ProtocolError("unknown sender or receiver")
    if message["sender"] not in _MESSAGE_SENDERS[message_type]:
        raise P3ProtocolError(f"{message_type} cannot be emitted by {message['sender']}")
    if not isinstance(message["seq"], int) or isinstance(message["seq"], bool) or message["seq"] < 0:
        raise P3ProtocolError("message.seq must be a non-negative integer")
    if message.get("diagnostic_code") is not None and message.get("diagnostic_code") not in DIAGNOSTIC_CODES:
        raise P3ProtocolError("unsupported diagnostic_code")
    if message.get("payload_ref") is not None:
        validate_artifact_ref(message["payload_ref"], field="message.payload_ref")
    output = dict(message)
    output["payload"] = validate_payload(message_type, message["payload"])
    return output


def make_message(*, run_id: str, message_id: str, message_type: str, sender: str,
                 receiver: str, seq: int, payload: Mapping[str, Any],
                 payload_ref: str | None = None, diagnostic_code: str = "OK") -> dict[str, Any]:
    message: dict[str, Any] = {
        "protocol": PROTOCOL, "run_id": run_id, "message_id": message_id,
        "type": message_type, "sender": sender, "receiver": receiver,
        "seq": seq, "payload": dict(payload), "diagnostic_code": diagnostic_code,
    }
    if payload_ref is not None:
        message["payload_ref"] = payload_ref
    return validate_message(message)


def dumps_message(message: Mapping[str, Any]) -> str:
    return json.dumps(validate_message(message), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads_message(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise P3ProtocolError("serialized message is not valid JSON") from exc
    return validate_message(result)


__all__ = [
    "AGENT_ROLES", "AgentMessage", "ArtifactStore", "DIAGNOSTIC_CODES", "MESSAGE_TYPES",
    "P3ProtocolError", "PROTOCOL", "PROTOCOL_SCHEMA", "dumps_message", "estimate_bytes",
    "estimate_message", "estimate_tokens", "loads_message", "make_message",
    "resolve_artifact_ref", "validate_artifact_ref", "validate_message", "validate_payload",
]
