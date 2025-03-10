import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

from . import SAVE_PATH

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_REPO_NAME = "Fosca709/dacon-hansol"


def get_model_dir(model_name: str) -> Path:
    return SAVE_PATH / "model" / model_name.replace("/", "--")


def download_snapshot(model_name: str, commit_hash: Optional[str] = None, **kwargs) -> None:
    local_dir = get_model_dir(model_name=model_name)
    snapshot_download(repo_id=model_name, revision=commit_hash, local_dir=local_dir, **kwargs)


def download_ko_sbert_sts() -> None:
    model_name = "jhgan/ko-sbert-sts"
    download_snapshot(model_name=model_name, ignore_patterns=["*.h5"])


def download_llama_varco() -> None:
    model_name = "NCSOFT/Llama-VARCO-8B-Instruct"
    commit_hash = "fe2d9358a2d35451c04e4589b47e361cfacd350d"
    download_snapshot(model_name=model_name, commit_hash=commit_hash)


def download_llama_rabbit() -> None:
    model_name = "CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct-2412"
    commit_hash = "ac6f1c0b756412163e17cb05d9e2f7ced274dc12"
    download_snapshot(model_name=model_name, commit_hash=commit_hash)


def get_date() -> str:
    return datetime.now(timezone.utc).strftime("%y%m%d_%H%M%S")


def get_save_name(run_name: str) -> str:
    return f"{get_date()}_{run_name}"


def hf_upload_folder(folder_path: Path) -> None:
    api = HfApi(token=HF_API_TOKEN)
    api.upload_folder(
        repo_id=HF_REPO_NAME,
        folder_path=folder_path,
        path_in_repo=folder_path.name,
    )


def hf_upload_file(file_path: Path, folder_in_repo: str = "") -> None:
    api = HfApi(token=HF_API_TOKEN)
    api.upload_file(path_or_fileobj=file_path, path_in_repo=f"{folder_in_repo}/{file_path.name}", repo_id=HF_REPO_NAME)
