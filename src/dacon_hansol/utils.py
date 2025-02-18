from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from . import SAVE_PATH


def get_model_dir(model_name: str) -> Path:
    return SAVE_PATH / "model" / model_name.replace("/", "--")


def download_snapshot(model_name: str, commit_hash: Optional[str] = None, **kwargs) -> None:
    local_dir = get_model_dir(model_name=model_name)
    snapshot_download(repo_id=model_name, revision=commit_hash, local_dir=local_dir, **kwargs)


def download_ko_sbert_sts() -> None:
    model_name = "jhgan/ko-sbert-sts"
    download_snapshot(model_name=model_name, ignore_patterns=["*.h5"])
