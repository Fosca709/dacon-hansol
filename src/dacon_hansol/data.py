import os
from typing import Literal

import numpy as np
import polars as pl
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM, apply_chat_template

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


DEFAULT_SYSTEM_PROMPT = """당신은 건설 안전 전문가입니다. 질문에 핵심 내용만 간략하게 답하세요. 서론, 배경 설명 또는 추가 설명 없이 바로 답변하세요."""
DEFAULT_QUESTION_PROMPT = "위와 같은 상황에서 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"


def get_user_prompt(text: str, question_prompt=DEFAULT_QUESTION_PROMPT):
    return f"{text}\n{question_prompt}"


def get_zero_shot_messages(text: str, system_prompt=DEFAULT_SYSTEM_PROMPT, question_prompt=DEFAULT_QUESTION_PROMPT):
    user_prompt = get_user_prompt(text, question_prompt=question_prompt)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return messages


def get_trl_dataset(df: pl.DataFrame, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    df = concat_fields(df)
    dataset = Dataset.from_polars(df)

    def make_template(row):
        messages = get_zero_shot_messages(row["text"])
        messages.append({"role": "assistant", "content": row["answer"]})
        return apply_chat_template({"messages": messages}, tokenizer)

    dataset = dataset.map(make_template)
    return dataset


def get_llama_collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorForCompletionOnlyLM:
    return DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, response_template="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def make_split_indices() -> np.ndarray:
    indices_path = DATA_PATH / "indices.npy"
    if os.path.exists(indices_path):
        return np.load(indices_path)

    TEST_SIZE = 500
    df = load_data("train")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=42)
    split = list(splitter.split(X=df, y=df["인적사고"]))
    val_indices = split[0][1]
    np.save(indices_path, val_indices)
    return val_indices


def train_val_split() -> tuple[pl.DataFrame, pl.DataFrame]:
    df = load_data("train")
    indices = make_split_indices()

    set_indices = set(indices)
    train_indices = [i for i in range(len(df)) if i not in set_indices]

    df_train = df[train_indices]
    df_val = df[indices]

    return df_train, df_val
