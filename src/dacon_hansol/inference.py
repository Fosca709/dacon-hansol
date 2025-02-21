import polars as pl
from transformers import GenerationMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .data import concat_fields, load_data

ZERO_SHOT_SYSTEM_PROMPT = """당신은 건설 안전 전문가입니다. 질문에 핵심 내용만 간략하게 답하세요. 서론, 배경 설명 또는 추가 설명 없이 바로 답변하세요."""
ZERO_SHOT_QUESTION_PROMPT = "\n위와 같은 상황에서 재발 방지 대책 및 향후 조치 계획은 무엇인가요?"


def get_zero_shot_messages(text: str, system_prompt=ZERO_SHOT_SYSTEM_PROMPT, question_prompt=ZERO_SHOT_QUESTION_PROMPT):
    user_prompt = f"{text}{question_prompt}"
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


def vllm_zero_shot(model, df: pl.DataFrame, max_new_tokens: int, **kwargs) -> pl.Series:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    df_processed = concat_fields(df)
    texts = df_processed["text"].to_list()
    conversations = [get_zero_shot_messages(text) for text in texts]

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=max_new_tokens, **kwargs)
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
        user_message = f"{row[0]}{question_prompt}"
        few_shot_messages.append({"role": "user", "content": user_message})

        assistant_message = row[1]
        few_shot_messages.append({"role": "assistant", "content": assistant_message})

    return few_shot_messages


def get_few_shot_messages(
    text: str, few_shot_messages: list[dict[str, str]], qeustion_prompt: str = ZERO_SHOT_QUESTION_PROMPT
) -> list[dict[str, str]]:
    messages = few_shot_messages.copy()
    messages.append({"role": "user", "content": f"{text}{qeustion_prompt}"})
    return messages


def vllm_few_shot(
    model,
    df: pl.DataFrame,
    few_shot_messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float = 0,
    **kwargs,
) -> pl.Series:
    from vllm import LLM, SamplingParams

    assert isinstance(model, LLM)

    df_processed = concat_fields(df)
    texts = df_processed["text"].to_list()
    conversations = [get_few_shot_messages(text, few_shot_messages) for text in texts]

    sampling_params = SamplingParams(n=1, temperature=temperature, max_tokens=max_new_tokens, **kwargs)
    outputs = model.chat(messages=conversations, sampling_params=sampling_params)
    output_texts = [output.outputs[0].text for output in outputs]
    return pl.Series(name="pred", values=output_texts)
