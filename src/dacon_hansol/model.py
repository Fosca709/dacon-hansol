import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import get_model_dir

MODEL_NAMES = {"varco": "NCSOFT/Llama-VARCO-8B-Instruct", "rabbit": "CarrotAI/Llama-3.2-Rabbit-Ko-3B-Instruct-2412"}


def load_ko_sbert_sts() -> SentenceTransformer:
    model_name = "jhgan/ko-sbert-sts"
    model_dir = get_model_dir(model_name)
    model = SentenceTransformer(model_dir.as_posix(), local_files_only=True)
    model.eval()
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    model_dir = get_model_dir(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    if model_name == MODEL_NAMES["varco"]:
        fix_varco_tokenizer(tokenizer)

    return tokenizer


def fix_varco_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    # chat template from rabbit's tokenizer
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Initially, pad_token_id is set to 0, which is the same ID as the "!" token.
    # Using pad_token_id for a commonly used token is not ideal.
    # Additionally, it does not align well with the data collator in the TRL library.
    # Therefore, it has been changed to the value used in rabbit's tokenizer.
    tokenizer.pad_token_id = 128001


def load_causal_model(model_name: str, torch_dtype: torch.dtype = torch.bfloat16, load_in_4bit: bool = False, **kwargs):
    model_dir = get_model_dir(model_name)

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, quantization_config=quantization_config, device_map="auto", **kwargs
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch_dtype, device_map="auto", **kwargs
        )

    return model


def load_vllm_chat_model(model_name: str):
    from vllm import LLM

    model_dir = get_model_dir(model_name)
    model = LLM(model=model_dir.as_posix(), max_model_len=8192, task="generate")
    return model


def load_mock_model(vocab_size: int = 128000) -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=vocab_size, hidden_size=128, intermediate_size=256, num_hidden_layers=1, num_attention_heads=2
    )

    return LlamaForCausalLM(config)


def load_unsloth_model(
    model_name: str,
    gpu_memory_utilization: float = 0.9,
    use_lora: bool = True,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    use_vllm: bool = False,
):
    from unsloth import FastLanguageModel

    model_dir = get_model_dir(model_name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_dir.as_posix(),
        load_in_4bit=load_in_4bit,
        gpu_memory_utilization=gpu_memory_utilization,
        fast_inference=use_vllm,
        random_state=42,
        local_files_only=True,
    )

    if model_name == MODEL_NAMES["varco"]:
        fix_varco_tokenizer(tokenizer)

    if use_lora:
        model = FastLanguageModel.get_peft_model(
            model=model, r=32, lora_alpha=64, random_state=42, max_seq_length=max_seq_length
        )

    return model, tokenizer


def save_peft_model(model, tokenizer, save_path) -> None:
    model.save_pretrained_merged(
        save_path,
        tokenizer,
        save_method="merged_16bit",
    )
