import os
from typing import Literal, Optional

import numpy as np
import polars as pl
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM, apply_chat_template

from . import SAVE_PATH
from .utils import download_reward_dataset

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


def get_grpo_dataset(df: pl.DataFrame) -> Dataset:
    df = concat_fields(df)
    dataset = Dataset.from_polars(df)

    def make_prompt(row):
        messages = get_zero_shot_messages(row["text"])
        return {"prompt": messages}

    dataset = dataset.map(make_prompt).remove_columns("text")
    return dataset


def fold_reward_dataframe(df_reward: pl.DataFrame, fixed_columns: list[str] | None = None) -> pl.DataFrame:
    columns_fixed = ["ID", "text"]
    if fixed_columns is not None:
        columns_fixed.extend(fixed_columns)
    columns_fold = [col for col in df_reward.columns if col not in columns_fixed]
    return df_reward.group_by(*columns_fixed, maintain_order=True).agg(*columns_fold)


def unfold_reward_dataframe(df_fold: pl.DataFrame, fixed_columns: list[str] | None = None) -> pl.DataFrame:
    columns_fixed = ["ID", "text"]
    if fixed_columns is not None:
        columns_fixed.extend(fixed_columns)
    columns_unfold = [col for col in df_fold.columns if col not in columns_fixed]
    return df_fold.explode(*columns_unfold)


def load_reward_dataset() -> pl.DataFrame:
    data_path = SAVE_PATH / "data" / "data" / "reward.parquet"
    if not os.path.exists(data_path):
        download_reward_dataset()

    return pl.read_parquet(data_path)


def train_val_split_for_reward(
    df_reward: pl.DataFrame,
    train_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_fold = fold_reward_dataframe(df_reward)

    indices = make_split_indices()
    set_indices = set(indices)
    train_indices = [i for i in range(len(df_fold)) if i not in set_indices]

    df_train = df_fold[train_indices]
    df_val = df_fold[indices]

    if train_size is not None:
        df_train = df_train.sample(n=train_size, seed=seed)
    if test_size is not None:
        df_val = df_val.sample(n=test_size, seed=seed)

    df_train = unfold_reward_dataframe(df_train)
    df_val = unfold_reward_dataframe(df_val)

    return df_train, df_val


def clean_reward_dataset(df_reward: pl.DataFrame) -> pl.DataFrame:
    """
    Remove duplicate answers and filter out IDs that have only a single remaining answer.
    """
    df = df_reward.unique(subset=["ID", "pred"], maintain_order=True)
    df = fold_reward_dataframe(df)
    mask = df["pred"].list.len() != 1
    df = df.filter(mask)
    return df
