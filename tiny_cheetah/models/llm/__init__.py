"""
LLM backends.

Default top-level exports remain tinygrad for backward compatibility when tinygrad is installed.
"""

try:
    from .tinygrad import *  # noqa: F401,F403
except ModuleNotFoundError as exc:
    if exc.name != "tinygrad":
        raise
