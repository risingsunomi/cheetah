from __future__ import annotations

__all__ = ["RepoHuggingFace", "RepoCustom"]


def __getattr__(name: str):
    if name == "RepoHuggingFace":
        from .repo_huggingface import RepoHuggingFace

        return RepoHuggingFace
    if name == "RepoCustom":
        from .repo_custom import RepoCustom

        return RepoCustom
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
