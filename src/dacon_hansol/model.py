from sentence_transformers import SentenceTransformer

from .utils import get_model_dir


def load_ko_sbert_sts() -> SentenceTransformer:
    model_name = "jhgan/ko-sbert-sts"
    model_dir = get_model_dir(model_name)
    model = SentenceTransformer(model_dir.as_posix(), local_files_only=True)
    model.eval()
    return model
