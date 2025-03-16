import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Self

import safetensors
import torch
import torch.nn as nn
from loguru import logger
from peft import PeftModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import SAVE_PATH
from .model import load_causal_model, load_tokenizer

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <bold>{message}</bold>",
    colorize=True,
)


class LastTokenPooling(nn.Module):
    def __init__(self, pad_token_id: int):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        sequence_lengths = torch.eq(input_ids, self.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        sequence_lengths = sequence_lengths.to(hidden_states.device)

        pooled = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
        return pooled


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = nn.Sequential(
            OrderedDict(
                [("fc1", nn.Linear(hidden_size, hidden_size)), ("act", nn.SiLU()), ("fc2", nn.Linear(hidden_size, 2))]
            )
        )

    def forward(self, hidden_states):
        if hidden_states.dtype != self.head.fc1.weight.dtype:
            hidden_states = hidden_states.to(self.head.fc1.weight.dtype)

        output = self.head(hidden_states)
        return output


class RewardModel(nn.Module):
    def __init__(self, causal_lm, tokenizer: PreTrainedTokenizerBase, value_head: ValueHead, is_peft: bool = False):
        super().__init__()
        self.causal_lm = causal_lm
        self.tokenizer = tokenizer

        self.pad_token_id = tokenizer.pad_token_id
        self.pooler = LastTokenPooling(pad_token_id=self.pad_token_id)

        self.value_head = value_head

        self.is_peft = is_peft

    def get_embed_model(self):
        if self.is_peft:
            return self.causal_lm.model.model
        return self.causal_lm.model

    def forward(self, input_ids, attention_mask):
        embed_model = self.get_embed_model()
        hidden_states = embed_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_states = self.pooler(hidden_states, input_ids)
        rewards = self.value_head(hidden_states)
        return rewards

    def save_model(self, model_name: str) -> None:
        model_path = SAVE_PATH / "model" / model_name

        if self.is_peft:
            self.causal_lm.save_pretrained(model_path)
            self.causal_lm.base_model.save_pretrained(model_path)
        else:
            self.causal_lm.save_pretrained(model_path)

        self.tokenizer.save_pretrained(model_path)

        safetensors.torch.save_model(self.value_head, filename=model_path / "v_head.safetensors")

    @classmethod
    def init_value_head(cls, model_path: Path, hidden_size: int) -> ValueHead:
        value_head = ValueHead(hidden_size)

        if os.path.exists(model_path / "v_head.safetensors"):
            v_head_state_dict = safetensors.torch.load_file(model_path / "v_head.safetensors")
            value_head.load_state_dict(v_head_state_dict)
            logger.info("Loaded v_head from the pretrained model path")

        else:
            logger.info("v_head not found. Initializing a new one")

        return value_head

    @classmethod
    def from_pretrained(cls, model_name: str) -> Self:
        model_path = SAVE_PATH / "model" / model_name
        is_peft = False

        logger.info("Loading causal_lm and tokenizer")
        causal_lm = load_causal_model(model_name)
        tokenizer = load_tokenizer(model_name)

        if os.path.exists(model_path / "adapter_model.safetensors"):
            logger.info("Adapter detected. Loading PEFT model")
            causal_lm = PeftModel.from_pretrained(causal_lm, model_path)
            is_peft = True

        hidden_size = causal_lm.config.hidden_size
        value_head = cls.init_value_head(model_path, hidden_size)
        value_head.to(causal_lm.device)

        return RewardModel(causal_lm=causal_lm, tokenizer=tokenizer, value_head=value_head, is_peft=is_peft)

    # import unsloth before calling this
    @classmethod
    def from_unsloth_pretrained(
        cls,
        model_name: str,
        gpu_memory_utilization: float = 0.9,
        use_lora: bool = False,
    ) -> Self:
        if "unsloth" not in sys.modules:
            logger.warning("It is recommended to import unsloth before calling this method")
        from .unsloth import load_unsloth_model

        causal_lm, tokenizer = load_unsloth_model(
            model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            use_lora=use_lora,
            load_in_4bit=False,
            use_vllm=False,
        )

        model_path = SAVE_PATH / "model" / model_name
        hidden_size = causal_lm.config.hidden_size
        value_head = cls.init_value_head(model_path, hidden_size)
        value_head.to(causal_lm.device)

        return RewardModel(causal_lm=causal_lm, tokenizer=tokenizer, value_head=value_head, is_peft=use_lora)
