import os
from typing import Literal, Optional

import chromadb
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import GenerationMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import SAVE_PATH
from .data import concat_fields, load_data
from .model import load_ko_sbert_sts

ZERO_SHOT_SYSTEM_PROMPT = """당신은 건설 안전 전문가입니다. 질문에 핵심 내용만 간략하게 답하세요. 서론, 배경 설명 또는 추가 설명 없이 바로 답변하세요."""
ZERO_SHOT_QUESTION_PROMPT = "위와 같은 상황에서 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
# ZERO_SHOT_QUESTION_PROMPT = "다음과 같은 상황에서 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"


def get_user_prompt(text: str, question_prompt=ZERO_SHOT_QUESTION_PROMPT):
    return f"{text}\n{question_prompt}"


def get_zero_shot_messages(text: str, system_prompt=ZERO_SHOT_SYSTEM_PROMPT, question_prompt=ZERO_SHOT_QUESTION_PROMPT):
    user_prompt = get_user_prompt(text, question_prompt=question_prompt)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return messages


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
    system_prompt: str = ZERO_SHOT_SYSTEM_PROMPT,
    question_prompt: str = ZERO_SHOT_QUESTION_PROMPT,
) -> list[dict[str, str]]:
    few_shot_messages = [{"role": "system", "content": system_prompt}]

    for row in df_examples.iter_rows():
        user_message = get_user_prompt(row[0], question_prompt=question_prompt)
        few_shot_messages.append({"role": "user", "content": user_message})

        assistant_message = row[1]
        few_shot_messages.append({"role": "assistant", "content": assistant_message})

    return few_shot_messages


def get_few_shot_messages(
    text: str, few_shot_messages: list[dict[str, str]], question_prompt: str = ZERO_SHOT_QUESTION_PROMPT
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
    system_prompt=ZERO_SHOT_SYSTEM_PROMPT,
    question_prompt=ZERO_SHOT_QUESTION_PROMPT,
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
