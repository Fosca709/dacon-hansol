import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


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
