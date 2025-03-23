import dataclasses
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Self

import numpy as np
import polars as pl
import safetensors
import torch
import torch.nn as nn
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from scipy.stats import kendalltau, spearmanr
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import SAVE_PATH
from .data import fold_reward_dataframe, get_zero_shot_messages, unfold_reward_dataframe
from .inference import make_multiple_zero_shot_samples
from .model import load_causal_model, load_tokenizer
from .optimizer import get_cosine_scheduler
from .utils import get_model_dir

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
                [("fc1", nn.Linear(hidden_size, hidden_size)), ("act", nn.SiLU()), ("fc2", nn.Linear(hidden_size, 1))]
            )
        )

    def forward(self, hidden_states):
        if hidden_states.dtype != self.head.fc1.weight.dtype:
            hidden_states = hidden_states.to(self.head.fc1.weight.dtype)

        output = self.head(hidden_states).squeeze(-1)
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

        self.set_gradient_flags()

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
    def from_pretrained(cls, model_name: str, base_model_name: Optional[str] = None) -> Self:
        model_path = SAVE_PATH / "model" / model_name
        is_peft = False

        if base_model_name is None:
            base_model_name = model_name

        logger.info("Loading causal_lm and tokenizer")
        causal_lm = load_causal_model(base_model_name)
        tokenizer = load_tokenizer(base_model_name)

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

    @property
    def device(self) -> torch.device:
        return self.causal_lm.device

    def set_gradient_flags(self) -> None:
        if self.is_peft:
            for name, param in self.named_parameters():
                if ("lora" in name) or ("value_head" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        else:
            for name, param in self.named_parameters():
                param.requires_grad = True


class PRORankingLoss(nn.Module):
    """
    Ranking loss inspired by 'Preference Ranking Optimization for Human Alignment (https://arxiv.org/abs/2306.17492)'.
    A more readable (but less efficient) implementation of this loss can be found in `pro_ranking_loss_readable`.
    """

    def __init__(self, scale=20.0, max_t=10.0):
        super().__init__()
        self.scale = scale
        self.max_t = max_t

    def forward(self, rewards: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # rewards, targets: 1D tensors sorted in decreasing order

        w = targets[:, None] - targets[None, :]

        diagonal_indices = torch.arange(targets.size(0))
        right_indices = (diagonal_indices + 1) % (targets.size(0))
        w[diagonal_indices, diagonal_indices] = w[diagonal_indices, right_indices]

        w = torch.clip(torch.reciprocal(self.scale * w), max=self.max_t, min=0.0)
        w = w * rewards[None, :]

        mask = torch.tril(torch.ones_like(w), diagonal=-1).bool()
        w.masked_fill_(mask, float("-inf"))

        log_softmax = torch.diag(torch.log_softmax(w, dim=1))
        return -log_softmax.sum()


def pro_ranking_loss_readable(rewards, targets, scale=20.0, max_t=10.0):
    return sum(
        [
            bt_loss_with_target_reward(rewards[i:], targets[i:], scale=scale, max_t=max_t)
            for i in range(len(rewards) - 1)
        ]
    )


def bt_loss_with_target_reward(rewards, targets, scale=20.0, max_t=10.0):
    t = targets[0] - targets
    t[0] = t[1]
    t = torch.clip(torch.reciprocal(scale * t), max=max_t)

    exp = torch.exp(t * rewards)
    return -torch.log(exp[0] / exp.sum())


def listmle(rewards, **kwargs):
    rewards = torch.flip(rewards, dims=(0,))
    return (torch.logcumsumexp(rewards, dim=0) - rewards).mean()


class ListMLE_with_MSE(nn.Module):
    def __init__(self, scale: float = 1.0, weight: float = 0.5):
        super().__init__()
        self.scale = scale
        self.weight = weight

    def forward(self, rewards: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ranking_loss = listmle(self.scale * rewards)
        mse_loss = torch.pow(rewards - targets, 2).mean()
        return ranking_loss + self.weight * mse_loss


def encode_reward_dataframe(df_reward: pl.DataFrame, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    def make_message(row):
        message = get_zero_shot_messages(row["text"])
        message.append({"role": "assistant", "content": row["pred"]})
        message = tokenizer.apply_chat_template(message, tokenize=False)
        return message

    df = unfold_reward_dataframe(df_reward)
    df = df.with_columns(pl.struct("text", "pred").map_elements(make_message, return_dtype=pl.String).alias("message"))
    df = fold_reward_dataframe(df)

    def tokenize(row):
        return tokenizer.batch_encode_plus(row["message"], padding=True)

    dataset = Dataset.from_polars(df.select("message", "score"))
    dataset = dataset.map(tokenize).remove_columns("message")
    return dataset


def get_logger(level: str, run_name: str) -> None:
    log_path = SAVE_PATH / "log" / run_name / f"{level}.log"
    if os.path.exists(log_path):
        raise Exception(f"{run_name}/{level}.log already existed")

    logger.level(name=level, no=8, color="<bold>")
    logger.add(
        log_path,
        level=level,
        format="{message}",
        filter=lambda record: record["level"].name == level,
    )

    logger.log(level, "step,loss,top1_diff,spearman,kendall")


def get_defaults_from_dataclass(cls) -> dict[str, Any]:
    defaults: dict[str, Any] = dict()
    for f in dataclasses.fields(cls):
        name = f.name
        if f.default != dataclasses.MISSING:
            defaults[name] = f.default

        elif f.default_factory != dataclasses.MISSING:
            defaults[name] = f.default_factory()

        else:
            raise Exception(f"Couldn't find a default value for <{name}>")

    return defaults


@dataclass
class RewardMetricLogger:
    loss: float = 0.0
    top1_diff: float = 0.0
    spearman: float = 0.0
    kendall: float = 0.0
    n_samples: int = 0

    def __init__(self, level: str, run_name: str):
        self.level = level
        self.run_name = run_name
        get_logger(level, run_name)
        self.step = 0

    def reset(self):
        defaults = get_defaults_from_dataclass(self.__class__)
        for name, value in defaults.items():
            setattr(self, name, value)

    def update_and_log(self, output: torch.Tensor, label: torch.Tensor, loss: float, log: bool = True) -> None:
        output = output.detach().cpu()
        label = label.detach().cpu()

        self.step += 1
        self.n_samples += 1

        # compute metrics
        spearman = np.nan_to_num(spearmanr(label, output).statistic, nan=0).item()
        kendall = np.nan_to_num(kendalltau(label, output).statistic, nan=0).item()
        top1_diff = (label.max() - label[output.argmax()]).item()

        # log metrics
        if log:
            logger.log(self.level, f"{self.step},{loss},{top1_diff},{spearman},{kendall}")

        # update metrics
        self.loss += loss
        self.top1_diff += top1_diff
        self.spearman += spearman
        self.kendall += kendall

    def get_metric(self) -> dict[str, float]:
        if self.n_samples == 0:
            raise Exception("No data for compute metrics")

        loss = self.loss / self.n_samples
        top1_dff = self.top1_diff / self.n_samples
        spearman = self.spearman / self.n_samples
        kendall = self.kendall / self.n_samples

        return {"loss": loss, "top1_diff": top1_dff, "spearman": spearman, "kendall": kendall}

    def print_log(self):
        metrics = self.get_metric()
        message_format = (
            "<{color}>{level} Metric on Step {step}:"
            "Loss {loss:.6f} | Top1 Diff {top1_diff:.6f} "
            "| Spearman {spearman:.6f} | Kendall {kendall:.6f}</{color}>"
        )
        color = "blue" if self.level == "train" else "red"
        message = message_format.format(
            color=color,
            level=self.level.upper(),
            step=self.step,
            loss=metrics["loss"],
            top1_diff=metrics["top1_diff"],
            spearman=metrics["spearman"],
            kendall=metrics["kendall"],
        )
        logger.opt(colors=True).info(message)


def train(
    run_name: str,
    model: RewardModel,
    train_dataset: Dataset,
    accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    print_steps: int = 10,
    seed: int = 42,
    loss_fn=None,
) -> None:
    model.train()

    train_dataset = train_dataset.shuffle(seed=seed)
    train_dataset.set_format("torch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_scheduler(
        optimizer=optimizer, training_steps=len(train_dataset) // accumulation_steps, warmup_ratio=warmup_ratio
    )

    if loss_fn is None:
        loss_fn = PRORankingLoss()

    train_logger = RewardMetricLogger(level="train", run_name=run_name)

    progress_bar = tqdm(total=len(train_dataset), desc="Train")
    for idx, batch in enumerate(train_dataset):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        score = batch["score"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Rearrange the data in descending order based on scores
        score_sorted = torch.sort(score, descending=True)
        score = score_sorted.values
        indices = score_sorted.indices
        input_ids = input_ids[indices]
        attention_mask = attention_mask[indices]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(rewards=output, targets=score)

        # log train metrics
        train_logger.update_and_log(output, score, loss.item(), log=True)
        if (idx + 1) % print_steps == 0:
            train_logger.print_log()
            train_logger.reset()

        loss = loss / accumulation_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        if (idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.refresh()

    progress_bar.close()


@torch.no_grad()
def validate(run_name: str, model: RewardModel, val_dataset: Dataset, loss_fn=None) -> None:
    model.eval()

    val_dataset.set_format("torch")

    val_logger = RewardMetricLogger(level="validation", run_name=run_name)

    if loss_fn is None:
        loss_fn = PRORankingLoss()

    progress_bar = tqdm(total=len(val_dataset), desc="Validation")
    for batch in val_dataset:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        score = batch["score"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Rearrange the data in descending order based on scores
        score_sorted = torch.sort(score, descending=True)
        score = score_sorted.values
        indices = score_sorted.indices
        input_ids = input_ids[indices]
        attention_mask = attention_mask[indices]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(rewards=output, targets=score)

        val_logger.update_and_log(output, score, loss.item(), log=True)

        progress_bar.update(1)
        progress_bar.refresh()

    progress_bar.close()
    val_logger.print_log()


def bon_generate(
    model,
    df: pl.DataFrame,
    num_generations: int,
    temperature: float = 1.5,
    top_p: float = 0.95,
    max_new_tokens=128,
    **sampling_kwargs,
) -> pl.DataFrame:
    df_generated = make_multiple_zero_shot_samples(
        model=model,
        df=df,
        num_generations=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        **sampling_kwargs,
    )
    df_generated.write_parquet(SAVE_PATH / "preds.parquet")
    return df_generated


def bon_predict_rewards(
    df_generated: pl.DataFrame,
    reward_model_name: str,
    base_model_name: Optional[str] = None,
    max_lora_rank: int = 32,
) -> pl.DataFrame:
    if base_model_name is None:
        base_model_name = reward_model_name
    base_model_path = get_model_dir(base_model_name)
    reward_model_path = get_model_dir(reward_model_name)

    tokenizer = load_tokenizer(base_model_name)

    fixed_column = None
    if "answer" in df_generated:
        fixed_column = ["answer"]
    logger.info("Process data")
    df_unfold = unfold_reward_dataframe(df_generated, fixed_columns=fixed_column)

    def make_message(row):
        message = get_zero_shot_messages(row["text"])
        message.append({"role": "assistant", "content": row["preds"]})

        return tokenizer.apply_chat_template(message, tokenize=False)

    df_messages = df_unfold.select(
        pl.struct("text", "preds").map_elements(make_message, return_dtype=pl.String).alias("message")
    )
    messages = df_messages["message"].to_list()

    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.lora.request import LoRARequest

    logger.info("Load base model on vllm")
    pooler_config = PoolerConfig(pooling_type="LAST", normalize=False, softmax=False)
    model = LLM(
        model=base_model_path.as_posix(),
        task="embed",
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        override_pooler_config=pooler_config,
    )
    lora_request = LoRARequest(lora_name=reward_model_name, lora_int_id=1, lora_path=reward_model_path.as_posix())
    outputs = model.embed(messages, lora_request=lora_request)
    embeddings = [out.outputs.embedding for out in outputs]

    logger.info("Load value head")
    hidden_size = len(embeddings[0])
    value_head = ValueHead(hidden_size=hidden_size)
    v_head_path = reward_model_path / "v_head.safetensors"
    v_head_state_dict = safetensors.torch.load_file(v_head_path)
    value_head.load_state_dict(v_head_state_dict)

    embeddings = torch.tensor(embeddings)
    with torch.no_grad():
        rewards = value_head(embeddings)
    rewards = rewards.numpy()

    pred_score = pl.Series(name="pred_score", values=rewards)
    df_unfold_with_score = df_unfold.with_columns(pred_score)
    df_with_score = fold_reward_dataframe(df_unfold_with_score, fixed_columns=fixed_column)
    df_with_score.write_parquet(SAVE_PATH / "preds_with_score.parquet")

    return df_with_score


def choose_best_preds(df_preds: pl.DataFrame) -> pl.Series:
    choice_expr = pl.col("pred_score").list.arg_max().alias("choice")

    def choice(row):
        return row["preds"][row["choice"]]

    df_pred = df_preds.select(
        pl.struct(pl.col("preds"), choice_expr).map_elements(choice, return_dtype=pl.String).alias("pred")
    )["pred"]
    return df_pred
