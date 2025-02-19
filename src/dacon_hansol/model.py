import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .utils import get_model_dir


def load_ko_sbert_sts() -> SentenceTransformer:
    model_name = "jhgan/ko-sbert-sts"
    model_dir = get_model_dir(model_name)
    model = SentenceTransformer(model_dir.as_posix(), local_files_only=True)
    model.eval()
    return model


def load_llama_varco_tokenizer() -> PreTrainedTokenizerBase:
    model_name = "NCSOFT/Llama-VARCO-8B-Instruct"
    model_dir = get_model_dir(model_name)
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def load_llama_varco_model(load_in_4bit: bool) -> LlamaForCausalLM:
    model_name = "NCSOFT/Llama-VARCO-8B-Instruct"
    model_dir = get_model_dir(model_name)

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, quantization_config=quantization_config, device_map="auto"
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto"
        )

    return model
