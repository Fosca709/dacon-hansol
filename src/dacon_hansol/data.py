from typing import Literal

import polars as pl

from . import SAVE_PATH

DATA_PATH = SAVE_PATH / "data"


def load_data(mode: Literal["train", "test", "submission"]) -> pl.DataFrame:
    if mode == "submission":
        return pl.read_csv(DATA_PATH / "sample_submission.csv")

    df = pl.read_csv(DATA_PATH / f"{mode}.csv")
    df = df.fill_null("-")
    return df


def concat_fields(df: pl.DataFrame) -> pl.DataFrame:
    ignore_columns = ["ID", "재발방지대책 및 향후조치계획"]
    columns = [x for x in df.columns if x not in ignore_columns]

    expr_add_prefix = [(pl.lit(f"{col}: ") + pl.col(col)).alias(col) for col in columns]
    expr_concat_str = [pl.concat_str(columns, separator="\n").alias("text")]
    if "재발방지대책 및 향후조치계획" in df.columns:
        expr_add_prefix.append(pl.col("재발방지대책 및 향후조치계획").alias("answer"))
        expr_concat_str.append(pl.col("answer"))

    df_processed = df.select(*expr_add_prefix).select(*expr_concat_str)
    return df_processed
