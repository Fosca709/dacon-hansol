from typing import Optional

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from . import SAVE_PATH
from .data import load_data
from .inference import vllm_zero_shot
from .model import MODEL_NAMES, load_ko_sbert_sts, load_vllm_chat_model


def make_embeddings(df_pred: pl.Series, model: Optional[SentenceTransformer] = None) -> np.ndarray:
    if model is None:
        model = load_ko_sbert_sts()

    embeddings = model.encode(df_pred.to_list(), show_progress_bar=True)
    return embeddings


def make_submission(df: pl.DataFrame, df_pred: pl.Series, embeddings: np.ndarray) -> pl.DataFrame:
    vec_fields = [f"vec_{i}" for i in range(768)]
    df_vec = pl.Series(name="vec", values=embeddings).arr.to_struct(fields=vec_fields).struct.unnest()
    df_sub = df.select(
        pl.col("ID"),
        df_pred.alias("재발방지대책 및 향후조치계획"),
    )
    df_sub = pl.concat([df_sub, df_vec], how="horizontal")
    return df_sub


def sample_submission() -> pl.DataFrame:
    df_test = load_data("test")
    df_pred = pl.select(pl.repeat("안전교육 실시", n=len(df_test)).alias("pred")).to_series()
    embeddings = make_embeddings(df_pred)
    submission = make_submission(df=df_test, df_pred=df_pred, embeddings=embeddings)
    submission.write_csv(SAVE_PATH / "submission.csv")
    return submission


def zero_shot_submission(
    model_name: str = "varco", max_new_tokens: int = 64, sample_size: Optional[int] = None
) -> None:
    model_name = MODEL_NAMES[model_name]
    model = load_vllm_chat_model(model_name)

    df_test = load_data("test")
    if sample_size is not None:
        df_test = df_test[:sample_size]

    df_pred = vllm_zero_shot(model=model, df=df_test, max_new_tokens=max_new_tokens)
    df_pred.to_frame().write_json(SAVE_PATH / "pred.json")

    embeddings = make_embeddings(df_pred=df_pred)
    submission = make_submission(df=df_test, df_pred=df_pred, embeddings=embeddings)
    submission.write_csv(SAVE_PATH / "submission.csv")
