"""Provider contract and source adapters."""

from .base import BaseProvider, ProviderError, ProviderResult
from .arxiv import ArxivProvider
from .local_library import FixtureLocalLibraryProvider, LocalLibraryProvider, LocalLibraryProviderProtocol

__all__ = [
    "ArxivProvider",
    "BaseProvider",
    "FixtureLocalLibraryProvider",
    "LocalLibraryProvider",
    "LocalLibraryProviderProtocol",
    "ProviderError",
    "ProviderResult",
]
