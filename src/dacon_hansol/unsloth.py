from unsloth import FastLanguageModel

from .model import MODEL_NAMES, fix_varco_tokenizer
from .utils import get_model_dir

"""
The `load_unsloth_model` function should be placed in a separate folder because Unsloth would be imported before Transformers and PEFT.
"""


def load_unsloth_model(
    model_name: str,
    gpu_memory_utilization: float = 0.9,
    use_lora: bool = True,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    use_vllm: bool = False,
):
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
