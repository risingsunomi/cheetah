"""
HuggingFace repo for downloading models
"""
from glob import glob
from pathlib import Path
from typing import Any

import huggingface_hub

from tiny_cheetah.models.llm.backend import backend_model_config_class

class RepoHuggingFace:
    def __init__(self, model_name: str, backend: str | None = None):
        self.model_name = model_name
        self.model_config = backend_model_config_class(backend=backend)()

    def download(self) -> tuple[Path, Any]:
        # TO DO: for now download whole model, later make it only download safetensors
        # and other model information it needs
        # use tqdm to display on textual frontend
        model_data = huggingface_hub.snapshot_download(repo_id=self.model_name)

        for file in glob(f"{model_data}/**/*", recursive=True):
            path = Path(file)
            if path.name == "config.json":
                print(f"Loading config from {file}")
                self.model_config.load(file)

            if path.name == "generation_config.json":
                print(f"Loading generation config from {file}")
                self.model_config.load_generation_config(Path(file))

        return Path(model_data), self.model_config
