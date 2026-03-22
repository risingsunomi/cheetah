from __future__ import annotations

import os
import re
from setuptools import setup, find_packages


def _normalize_cuda_tag(raw_variant: str) -> str | None:
    variant = raw_variant.strip().lower()
    if variant in {"", "auto", "cpu", "mps", "default"}:
        return None

    if "rocm" in variant:
        raise RuntimeError(
            "ROCm variants are no longer supported by this project installer. "
            "Use CPU/MPS or CUDA (e.g. cu126, cu128, cu130)."
        )

    # Accept common formats:
    # cu126, cu12.6, cuda12.6, cuda-12.8, 12.6, 130
    direct = re.fullmatch(r"cu(\d{3,4})", variant)
    if direct:
        return f"cu{direct.group(1)}"

    dotted = re.fullmatch(r"(?:cu|cuda)?[-_ ]?(\d{1,2})[._](\d{1,2})", variant)
    if dotted:
        major = int(dotted.group(1))
        minor = int(dotted.group(2))
        return f"cu{major}{minor}"

    plain = re.fullmatch(r"(\d{3,4})", variant)
    if plain:
        return f"cu{plain.group(1)}"

    return None


def _torch_requirement() -> str:
    """
    Resolve torch dependency from environment.

    Supported env vars:
    - TC_TORCH_VARIANT: auto|cpu|mps|cu126|cu12.6|cuda12.8|cu130|13.0|...
    - TC_TORCH_VERSION: base torch version (default: 2.5.1)

    Notes:
    - CUDA local-version wheels (e.g. +cu126) generally require the matching
      PyTorch wheel index to be configured in pip.
    """
    variant = os.getenv("TC_TORCH_VARIANT", "auto").strip().lower()
    version = os.getenv("TC_TORCH_VERSION", "2.10.0").strip()

    # Default resolver path (PyPI / platform default wheel).
    tag = _normalize_cuda_tag(variant)
    if tag is None:
        return f"torch>={version}"
    return f"torch=={version}+{tag}"


BASE_REQUIRES = [
    "tinygrad",
    "numpy",
    "requests",
    "safetensors",
    "textual",
    "transformers",
    "jinja2",
    "huggingface-hub",
    "python-dotenv",
    "tokenizers",
    "zstandard",
    _torch_requirement(),
]

_DEFAULT_TORCH_VERSION = os.getenv("TC_TORCH_VERSION", "2.5.1").strip()

EXTRAS = {
    "dev": [
        "pytest",
        "textual-dev",
    ],
    # Use with pip wheel index URLs if selecting CUDA variants.
    "torch-cpu": [f"torch>={_DEFAULT_TORCH_VERSION}"],
    "torch-cu126": [f"torch=={_DEFAULT_TORCH_VERSION}+cu126"],
    "torch-cu128": [f"torch=={_DEFAULT_TORCH_VERSION}+cu128"],
    "torch-cu130": [f"torch=={_DEFAULT_TORCH_VERSION}+cu130"],
}

setup(
    name="cheetah",
    version="0.1",
    description="Distributed inference and training with tinygrad or torch backends",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "tiny_cheetah.agent": ["functions.json"],
        "tiny_cheetah.agent.prompts": ["*.j2"],
        "tiny_cheetah.tui": ["*.tcss"],
        "tiny_cheetah.tui.widget": ["*.tcss"],
    },
    python_requires=">=3.10",
    install_requires=BASE_REQUIRES,
    extras_require=EXTRAS,
)
