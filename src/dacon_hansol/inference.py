import polars as pl
from transformers import GenerationMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .data import concat_fields

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
