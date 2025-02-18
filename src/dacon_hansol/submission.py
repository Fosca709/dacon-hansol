from typing import Optional

import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from . import SAVE_PATH
from .model import load_ko_sbert_sts


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
    df_test = pl.read_csv(SAVE_PATH / "data" / "test.csv")
    df_pred = pl.select(pl.repeat("안전교육 실시", n=len(df_test)).alias("pred")).to_series()
    embeddings = make_embeddings(df_pred)
    submission = make_submission(df=df_test, df_pred=df_pred, embeddings=embeddings)
    submission.write_csv(SAVE_PATH / "submission.csv")
    return submission
