"""P1 本地配置读取。

只读取根目录 ``.env.local`` 和进程环境变量；环境变量优先。该模块不做
网络请求，也不会在表示配置时回显密钥。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = ROOT_DIR / ".env.local"

DEFAULTS: dict[str, str] = {
    "BOHRIUM_BASE_URL": "https://open.bohrium.com",
    "BOHRIUM_ACCOUNT": "",
    "BOHR_ACCESS_KEY": "",
    "BOHRIUM_PASSWORD": "",
    "OPENALEX_BASE_URL": "https://api.openalex.org",
    "OPENALEX_ACCOUNT": "",
    "OPENALEX_API_KEY": "",
    "OPENALEX_PASSWORD": "",
    "OPENALEX_MAILTO": "",
    "SCIVERSE_API_BASE_URL": "https://api.sciverse.space",
    "SCIVERSE_API_TOKEN": "",
}

_SENSITIVE_RE = re.compile(r"(?:KEY|TOKEN|PASSWORD|SECRET|AUTH|CREDENTIAL|EMAIL|MAILTO)", re.I)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].lstrip()
    if "=" not in line:
        return None
    name, value = line.split("=", 1)
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return name, value


def read_env_file(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """读取简单 dotenv 文件；不存在时返回空字典。"""

    env_path = Path(path) if path is not None else DEFAULT_ENV_FILE
    if not env_path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed:
            values[parsed[0]] = parsed[1]
    return values


def load_config(
    env_file: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """合并默认值、根 ``.env.local`` 和进程环境变量。

    ``environ`` 主要用于测试；传入时不会读取当前进程环境之外的变量。
    未列入默认值的变量也会保留，便于后续 Provider 扩展。
    """

    values = dict(DEFAULTS)
    values.update(read_env_file(env_file))
    process_env = os.environ if environ is None else environ
    values.update({key: value for key, value in process_env.items() if isinstance(value, str)})
    return values


@dataclass(frozen=True)
class ProviderSettings:
    """Provider 的非敏感连接信息和认证状态。"""

    name: str
    base_url: str
    api_key: str = ""
    access_key: str = ""
    account: str = ""
    password: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.api_key or self.access_key or self.account)

    def redacted(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "api_key": redact_value(self.api_key),
            "access_key": redact_value(self.access_key),
            "account": redact_value(self.account),
            "password": redact_value(self.password),
            "configured": self.configured,
        }


def get_provider_config(config: Mapping[str, str], provider: str) -> ProviderSettings:
    """从配置映射构造标准 ProviderSettings。"""

    name = provider.casefold().replace("_", "-")
    if name in {"bohr", "bohrium"}:
        return ProviderSettings(
            "bohrium",
            config.get("BOHRIUM_BASE_URL", DEFAULTS["BOHRIUM_BASE_URL"]),
            access_key=config.get("BOHR_ACCESS_KEY", ""),
            account=config.get("BOHRIUM_ACCOUNT", ""),
            password=config.get("BOHRIUM_PASSWORD", ""),
        )
    if name == "openalex":
        return ProviderSettings(
            "openalex",
            config.get("OPENALEX_BASE_URL", DEFAULTS["OPENALEX_BASE_URL"]),
            api_key=config.get("OPENALEX_API_KEY", ""),
            account=config.get("OPENALEX_ACCOUNT", ""),
            password=config.get("OPENALEX_PASSWORD", ""),
        )
    if name == "sciverse":
        return ProviderSettings(
            "sciverse",
            config.get("SCIVERSE_API_BASE_URL", DEFAULTS["SCIVERSE_API_BASE_URL"]),
            api_key=config.get("SCIVERSE_API_TOKEN", ""),
        )
    raise KeyError(f"unknown provider: {provider}")


def redact_value(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def redact_config(config: Mapping[str, str]) -> dict[str, str]:
    """返回可安全写入日志的配置副本。"""

    return {
        key: redact_value(value) if _SENSITIVE_RE.search(key) else value
        for key, value in config.items()
    }


def redact_url(url: str) -> str:
    """隐藏 URL 查询参数中的 key/token/password 等敏感值。"""

    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "***" if _SENSITIVE_RE.search(key) else value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


__all__ = [
    "DEFAULT_ENV_FILE",
    "DEFAULTS",
    "ProviderSettings",
    "get_provider_config",
    "load_config",
    "read_env_file",
    "redact_config",
    "redact_url",
    "redact_value",
]
