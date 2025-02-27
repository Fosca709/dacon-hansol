import os
from typing import Literal, Optional

import chromadb
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import GenerationMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import SAVE_PATH
from .data import (
    DEFAULT_QUESTION_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    concat_fields,
    get_user_prompt,
    get_zero_shot_messages,
    load_data,
)
from .model import load_ko_sbert_sts


def naive_zero_shot(
    model: GenerationMixin, tokenizer: PreTrainedTokenizerBase, max_new_tokens: int, text: str, **kwargs
):
    messages = get_zero_shot_messages(text)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

    outputs = model.generate(
        inputs,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )
    return outputs


def get_default_sampling_params(max_new_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(n=1, temperature=0.5, top_p=0.9, max_tokens=max_new_tokens)


def vllm_zero_shot(model, df: pl.DataFrame, max_new_tokens: int, sampling_params=None) -> pl.Series:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    df_processed = concat_fields(df)
    texts = df_processed["text"].to_list()
    conversations = [get_zero_shot_messages(text) for text in texts]

    if sampling_params is None:
        sampling_params = get_default_sampling_params(max_new_tokens)
    assert isinstance(sampling_params, SamplingParams)

    outputs = model.chat(messages=conversations, sampling_params=sampling_params)
    output_texts = [output.outputs[0].text for output in outputs]
    return pl.Series(name="pred", values=output_texts)


def get_default_few_shot_examples(num_examples: int) -> pl.DataFrame:
    indices = [6162, 9414, 1228, 4681, 12920, 19761, 22766, 713]
    assert num_examples <= len(indices)

    df_train = load_data("train")
    df_processed = concat_fields(df_train)
    return df_processed[indices[:num_examples]]


def get_messages_from_frame(
    df_examples: pl.DataFrame,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    question_prompt: str = DEFAULT_QUESTION_PROMPT,
) -> list[dict[str, str]]:
    few_shot_messages = [{"role": "system", "content": system_prompt}]

    for row in df_examples.iter_rows():
        user_message = get_user_prompt(row[0], question_prompt=question_prompt)
        few_shot_messages.append({"role": "user", "content": user_message})

        assistant_message = row[1]
        few_shot_messages.append({"role": "assistant", "content": assistant_message})

    return few_shot_messages


def get_few_shot_messages(
    text: str, few_shot_messages: list[dict[str, str]], question_prompt: str = DEFAULT_QUESTION_PROMPT
) -> list[dict[str, str]]:
    messages = few_shot_messages.copy()
    user_message = get_user_prompt(text, question_prompt=question_prompt)
    messages.append({"role": "user", "content": user_message})
    return messages


def vllm_few_shot(
    model,
    df: pl.DataFrame,
    few_shot_messages: list[dict[str, str]],
    max_new_tokens: int,
    sampling_params=None,
) -> pl.Series:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    df_processed = concat_fields(df)
    texts = df_processed["text"].to_list()
    conversations = [get_few_shot_messages(text, few_shot_messages) for text in texts]

    if sampling_params is None:
        sampling_params = get_default_sampling_params(max_new_tokens)
    assert isinstance(sampling_params, SamplingParams)

    outputs = model.chat(messages=conversations, sampling_params=sampling_params)
    output_texts = [output.outputs[0].text for output in outputs]
    return pl.Series(name="pred", values=output_texts)


def load_data_with_embed(
    mode: Literal["train", "test"], embed_model: Optional[SentenceTransformer] = None
) -> pl.DataFrame:
    file_path = SAVE_PATH / "data" / f"{mode}_with_embed.parquet"
    if os.path.exists(file_path):
        df = pl.read_parquet(file_path)
        return df

    if embed_model is None:
        embed_model = load_ko_sbert_sts()

    df = load_data(mode)
    df_cat = concat_fields(df)

    texts_with_question = df_cat["text"].map_elements(get_user_prompt, return_dtype=pl.String).to_list()
    embeddings = embed_model.encode(texts_with_question, show_progress_bar=True)
    df_embed = pl.Series(name="embedding", values=embeddings)

    df_with_embed = df_cat.select(df["ID"], pl.all(), df_embed)
    df_with_embed.write_parquet(file_path)
    return df_with_embed


def get_rag_conversations(
    num_retrievals: int,
    train_sample_size: Optional[int] = None,
    test_sample_size: Optional[int] = None,
    batch_size: int = 1024,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    question_prompt=DEFAULT_QUESTION_PROMPT,
    embed_model: Optional[SentenceTransformer] = None,
) -> list[dict[str, str]]:
    df_train = load_data_with_embed("train", embed_model=embed_model)
    if train_sample_size is not None:
        df_train = df_train[:train_sample_size]

    df_test = load_data_with_embed("test", embed_model=embed_model)
    if test_sample_size is not None:
        df_test = df_test[:test_sample_size]

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="train", metadata={"hnsw:space": "cosine"})

    idx = 0
    while idx < len(df_train):
        df_batch = df_train[idx : idx + batch_size]

        ids = df_batch["ID"].to_list()
        embeddings = df_batch["embedding"].to_list()
        metadatas = df_batch.select(pl.struct("text", "answer").alias("meta"))["meta"].to_list()

        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

        idx += batch_size

    conversations = []
    for row in tqdm(df_test.iter_rows(named=True)):
        text = row["text"]
        embedding = row["embedding"]
        retrieved = collection.query(query_embeddings=embedding, n_results=num_retrievals)
        retrieved = retrieved["metadatas"][0]

        messages = [{"role": "system", "content": system_prompt}]
        for r in retrieved:
            messages.append({"role": "user", "content": get_user_prompt(r["text"], question_prompt)})
            messages.append({"role": "assistant", "content": r["answer"]})
        messages.append({"role": "user", "content": get_user_prompt(text, question_prompt)})
        conversations.append(messages)

    return conversations


def vllm_few_shot_with_rag(
    model,
    num_retrievals: int,
    max_new_tokens: int,
    train_sample_size: Optional[int] = None,
    test_sample_size: Optional[int] = None,
    embed_model: Optional[SentenceTransformer] = None,
    sampling_params=None,
) -> pl.Series:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    conversations = get_rag_conversations(
        num_retrievals=num_retrievals,
        train_sample_size=train_sample_size,
        test_sample_size=test_sample_size,
        embed_model=embed_model,
    )

    if sampling_params is None:
        sampling_params = get_default_sampling_params(max_new_tokens)
    assert isinstance(sampling_params, SamplingParams)

    outputs = model.chat(messages=conversations, sampling_params=sampling_params)
    output_texts = [output.outputs[0].text for output in outputs]
    return pl.Series(name="pred", values=output_texts)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-8) -> np.ndarray:
    # a,b : [B, N], output: [B,]
    dot_product = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b + eps)


def jaccard_similarity(text1, text2):
    """자카드 유사도 계산"""
    set1, set2 = set(text1.split()), set(text2.split())  # 단어 집합 생성
    intersection = len(set1.intersection(set2))  # 교집합 크기
    union = len(set1.union(set2))  # 합집합 크기
    return intersection / union if union != 0 else 0


def make_scores(df: pl.DataFrame, df_pred: pl.Series, embed_model: SentenceTransformer) -> pl.DataFrame:
    df_score = df.with_columns(df_pred)

    def compute_jaccard(row):
        return jaccard_similarity(row["answer"], row["pred"])

    df_score = df_score.with_columns(
        pl.struct("answer", "pred").alias("jaccard").map_elements(compute_jaccard, return_dtype=pl.Float64)
    )

    answers_embed = embed_model.encode(df_score["answer"].to_list(), show_progress_bar=True)
    preds_embed = embed_model.encode(df_score["pred"].to_list(), show_progress_bar=True)
    cos_scores = cosine_similarity(answers_embed, preds_embed)
    df_score = df_score.with_columns(pl.Series(name="cosine", values=cos_scores))

    df_score = df_score.with_columns(
        (0.7 * pl.col("cosine").clip(lower_bound=0.0) + 0.3 * pl.col("jaccard").clip(lower_bound=0.0)).alias("score")
    )
    return df_score


def validate(
    model,
    df: pl.DataFrame,
    embed_model: Optional[SentenceTransformer] = None,
    max_new_tokens: int = 128,
    sampling_params=None,
) -> pl.DataFrame:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    if sampling_params is None:
        sampling_params = SamplingParams(n=1, max_tokens=max_new_tokens, temperature=0)

    df_pred = vllm_zero_shot(model=model, df=df, max_new_tokens=max_new_tokens, sampling_params=sampling_params)
    df_cat = concat_fields(df)

    if embed_model is None:
        embed_model = load_ko_sbert_sts()
    df_score = make_scores(df=df_cat, df_pred=df_pred, embed_model=embed_model)

    print(f"Average score: {df_score['score'].mean()}")
    return df_score
