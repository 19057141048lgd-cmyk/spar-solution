"""Local paper-library provider for P1 fixture/configuration experiments."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from ..paperdoc import validate_paper_doc
from .base import ProviderError, ProviderResult


@runtime_checkable
class LocalLibraryProviderProtocol(Protocol):
    name: str
    library_status: str

    def search(self, query: str, *, page_size: int = 10, cursor: str | None = None, **kwargs: Any) -> ProviderResult:
        ...


def _query_records(records: Any, query: str) -> list[Any]:
    if isinstance(records, Mapping):
        exact = records.get(query)
        if isinstance(exact, list):
            return exact
        default = records.get("*")
        return default if isinstance(default, list) else []
    return list(records) if isinstance(records, list) else []


class LocalLibraryProvider:
    """Read PaperDoc records from a fixture or an explicitly configured JSON path.

    ``library_status`` is always one of ``mock``, ``configured`` or
    ``unavailable`` and is copied into every ProviderResult provenance object.
    """

    name = "local_library"

    @property
    def availability_status(self) -> str:
        return self.library_status

    def __init__(self, records: Any = None, *, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        if records is not None and path is not None:
            raise ProviderError(self.name, "config", "records and path are mutually exclusive")
        self._records = deepcopy(records)
        self.library_status = "mock" if records is not None else "configured" if self.path and self.path.is_file() else "unavailable"

    @property
    def status(self) -> str:
        """Compatibility alias used by experiment manifests."""

        return self.library_status

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "LocalLibraryProvider":
        value = (config or {}).get("LOCAL_LIBRARY_PATH") or (config or {}).get("LOCAL_LIBRARY_FILE")
        return cls(path=value) if value else cls()

    def _load(self) -> Any:
        if self.library_status == "unavailable":
            raise ProviderError(self.name, "config_missing", "local library path is not configured", details={"library_status": "unavailable"})
        if self.library_status == "mock":
            return self._records
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProviderError(self.name, "config_missing", "configured local library path does not exist", details={"library_status": "unavailable"}) from exc
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError(self.name, "parse", "local library JSON could not be read", details={"library_status": "configured"}) from exc
        return payload.get("papers", payload.get("records", payload)) if isinstance(payload, Mapping) else payload

    def search(self, query: str, *, page_size: int = 10, cursor: str | None = None, **_: Any) -> ProviderResult:
        if not isinstance(query, str) or not query.strip():
            raise ProviderError(self.name, "config", "query must be a non-empty string")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 200:
            raise ProviderError(self.name, "config", "page_size must be an integer between 1 and 200")
        raw_records = _query_records(self._load(), query.strip())
        records: list[dict[str, Any]] = []
        for index, record in enumerate(raw_records[:page_size]):
            if not isinstance(record, Mapping):
                raise ProviderError(self.name, "parse", f"records[{index}] must be an object")
            try:
                records.append(validate_paper_doc(deepcopy(dict(record))))
            except Exception as exc:
                raise ProviderError(self.name, "parse", f"records[{index}] is not a valid PaperDoc") from exc
        return ProviderResult(
            self.name,
            "search",
            records,
            total=len(raw_records),
            provenance={"endpoint": str(self.path) if self.path else "fixture://local-library", "library_status": self.library_status},
        )

    def read(self, paper_id: str, *, cursor: str | None = None, **_: Any) -> ProviderResult:
        raise ProviderError(self.name, "unsupported", "local library read is not configured")

    def relations(self, paper_id: str, *, relation: str = "all", cursor: str | None = None, **_: Any) -> ProviderResult:
        raise ProviderError(self.name, "unsupported", "local library relations are not configured")


class FixtureLocalLibraryProvider(LocalLibraryProvider):
    """Explicit name for mock-only local-library runs."""

    def __init__(self, records: Any) -> None:
        super().__init__(records)


JsonLocalLibraryProvider = LocalLibraryProvider


__all__ = ["FixtureLocalLibraryProvider", "JsonLocalLibraryProvider", "LocalLibraryProvider", "LocalLibraryProviderProtocol"]
