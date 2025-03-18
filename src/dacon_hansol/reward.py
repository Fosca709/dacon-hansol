import dataclasses
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Self

import numpy as np
import polars as pl
import safetensors
import torch
import torch.nn as nn
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from . import SAVE_PATH
from .data import get_zero_shot_messages
from .model import load_causal_model, load_tokenizer
from .optimizer import get_cosine_scheduler

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


def encode_reward_dataframe(df_reward: pl.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pl.DataFrame:
    def tokenize(row):
        message = get_zero_shot_messages(row["text"])
        message.append({"role": "assistant", "content": row["pred"]})
        input_ids = tokenizer.apply_chat_template(message)
        return input_ids

    input_ids_expr = pl.struct("text", "pred").map_elements(tokenize, return_dtype=pl.List(pl.Int64)).alias("input_ids")
    label_expr = pl.concat_list(pl.col("cosine", "jaccard").cast(pl.List(pl.Float32))).alias("label")

    return df_reward.select(input_ids_expr, label_expr)


class RewardCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features) -> dict[str, torch.Tensor]:
        return self.tokenizer.pad(encoded_inputs=features, padding=True, return_tensors="pt")


def get_reward_dataloader(
    df_encoded: pl.DataFrame, tokenizer: PreTrainedTokenizerBase, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = Dataset.from_polars(df_encoded)
    collator = RewardCollator(tokenizer)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=(0.49, 0.09)):
        super().__init__()
        self.weights = weights

    def forward(self, input, target):
        # input, target: [B, 2]
        w1, w2 = self.weights
        squared_errors = torch.pow(input - target, 2)
        return w1 * squared_errors[:, 0].mean() + w2 * squared_errors[:, 1].mean()


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

    logger.log(level, "step,loss,cosine_mse,jaccard_mse,score_mse")


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
    cosine_mse: float = 0.0
    jaccard_mse: float = 0.0
    score_mse: float = 0.0
    n_samples: int = 0

    def __init__(self, level: str, run_name: str, weights=(0.49, 0.09)):
        self.level = level
        self.run_name = run_name
        get_logger(level, run_name)

        self.weights = weights

        self.step = 0
        self.bs = 0
        self.cos = 0.0
        self.jac = 0.0
        self.sc = 0.0

    def reset(self):
        defaults = get_defaults_from_dataclass(self.__class__)
        for name, value in defaults.items():
            setattr(self, name, value)

    def compute_step(self, output: torch.Tensor, label: torch.Tensor) -> None:
        output = output.detach().cpu()
        label = label.detach().cpu()

        self.step += 1
        self.bs = output.shape[0]

        squared_error = torch.pow(output - label, 2)
        self.cos = squared_error[:, 0].sum().item()
        self.jac = squared_error[:, 1].sum().item()

        pred_reward = 0.7 * output[:, 0] + 0.3 * output[:, 1]
        target_reward = 0.7 * label[:, 0] + 0.3 * label[:, 1]
        self.sc = torch.sum(torch.pow(pred_reward - target_reward, 2)).item()

    def update(self) -> None:
        self.n_samples += self.bs
        self.cosine_mse += self.cos
        self.jaccard_mse += self.jac

        w1, w2 = self.weights
        self.loss += w1 * self.cos + w2 * self.jac

        self.score_mse += self.sc

    def get_metric(self) -> dict[str, float]:
        if self.n_samples == 0:
            raise Exception("No data for compute metrics")

        loss = self.loss / self.n_samples
        cosine_mse = self.cosine_mse / self.n_samples
        jaccard_mse = self.jaccard_mse / self.n_samples
        score_mse = self.score_mse / self.n_samples

        return {"loss": loss, "cosine_mse": cosine_mse, "jaccard_mse": jaccard_mse, "score_mse": score_mse}

    def log(self):
        step = self.step
        cosine_mse = self.cos / self.bs
        jaccard_mse = self.jac / self.bs

        w1, w2 = self.weights
        loss = w1 * cosine_mse + w2 * jaccard_mse

        score_mse = self.sc / self.bs

        logger.log(self.level, f"{step},{loss},{cosine_mse},{jaccard_mse},{score_mse}")

    def print_log(self):
        metrics = self.get_metric()
        message_format = (
            "<{color}>{level} Metric on Step {step}:"
            "Loss {loss:.6f} | Cosine MSE {cosine:.6f} "
            "| Jaccard MSE {jaccard:.6f} | Score MSE {score:.6f}</{color}>"
        )
        color = "blue" if self.level == "train" else "red"
        message = message_format.format(
            color=color,
            level=self.level.upper(),
            step=self.step,
            loss=metrics["loss"],
            cosine=metrics["cosine_mse"],
            jaccard=metrics["jaccard_mse"],
            score=metrics["score_mse"],
        )
        logger.opt(colors=True).info(message)


def train(
    run_name: str,
    model: RewardModel,
    df_train: pl.DataFrame,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    warmup_ratio: float = 0.1,
    loss_weights: tuple[float, float] = (0.49, 0.09),
    max_grad_norm: float = 1.0,
    print_steps: int = 10,
) -> None:
    model.train()
    tokenizer = model.tokenizer

    df_train_encoded = encode_reward_dataframe(df_train, tokenizer)
    train_dataloader = get_reward_dataloader(df_train_encoded, tokenizer, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_cosine_scheduler(
        optimizer=optimizer, training_steps=len(train_dataloader), warmup_ratio=warmup_ratio
    )

    loss_fn = WeightedMSELoss(weights=loss_weights)

    train_logger = RewardMetricLogger(level="train", run_name=run_name, weights=loss_weights)

    progress_bar = tqdm(total=len(train_dataloader), desc="Train")
    for idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        label = batch["label"]
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        # log train metrics
        train_logger.compute_step(output, label)
        train_logger.update()
        train_logger.log()
        if (idx + 1) % print_steps == 0:
            train_logger.print_log()
            train_logger.reset()

        loss = loss_fn(output, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        progress_bar.refresh()

    progress_bar.close()


@torch.no_grad()
def validate(
    run_name: str, model: RewardModel, df_val: pl.DataFrame, batch_size: int = 4, loss_weights=(0.49, 0.09)
) -> None:
    model.eval()
    tokenizer = model.tokenizer

    df_val_encoded = encode_reward_dataframe(df_val, tokenizer)
    val_dataloader = get_reward_dataloader(df_val_encoded, tokenizer, batch_size=batch_size, shuffle=False)

    val_logger = RewardMetricLogger(level="validation", run_name=run_name, weights=loss_weights)

    preds = []

    progress_bar = tqdm(total=len(val_dataloader), desc="Validation")
    for batch in val_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        label = batch["label"]
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

        preds.append(output.detach().cpu().numpy())

        val_logger.compute_step(output, label)
        val_logger.update()

        progress_bar.update(1)
        progress_bar.refresh()

    progress_bar.close()
    val_logger.print_log()

    preds = np.vstack(preds).clip(min=0, max=1)
    pred_cosine = preds[:, 0]
    pred_jaccard = preds[:, 1]
    pred_score = 0.7 * pred_cosine + 0.3 * pred_jaccard

    df_val_with_preds = df_val.with_columns(
        pl.Series(name="pred_cosine", values=pred_cosine),
        pl.Series(name="pred_jaccard", values=pred_jaccard),
        pl.Series(name="pred_score", values=pred_score),
    )

    df_rank = compute_rank_correlations(df_val_with_preds)
    df_rank.write_parquet(SAVE_PATH / "log" / run_name / "val_rank.parquet")

    message_format = (
        "<{color}>Rank Correlation:\n"
        "Cosine Spearman: {cs:.4f} Kendall: {ck:.4f}\n"
        "Jaccard Spearman: {js:.4f} Kendall: {jk:.4f}\n"
        "Score Spearman: {ss:.4f} Kendall: {sk:.4f}"
        "</{color}>"
    )
    message = message_format.format(
        color="red",
        cs=df_rank["cosine_spearman"].mean(),
        ck=df_rank["cosine_kendall"].mean(),
        js=df_rank["jaccard_spearman"].mean(),
        jk=df_rank["jaccard_kendall"].mean(),
        ss=df_rank["score_spearman"].mean(),
        sk=df_rank["score_kendall"].mean(),
    )
    logger.opt(colors=True).info(message)


def fold_reward_dataframe(df_reward: pl.DataFrame) -> pl.DataFrame:
    columns_fixed = ["ID", "text"]
    columns_fold = [col for col in df_reward.columns if col not in columns_fixed]
    return df_reward.group_by(*columns_fixed, maintain_order=True).agg(*columns_fold)


def unfold_reward_dataframe(df_fold: pl.DataFrame) -> pl.DataFrame:
    columns_fixed = ["ID", "text"]
    columns_unfold = [col for col in df_fold.columns if col not in columns_fixed]
    return df_fold.explode(*columns_unfold)


def compute_rank_correlations(df_with_preds: pl.DataFrame) -> pl.DataFrame:
    df_fold = fold_reward_dataframe(df_with_preds)

    def _map(row, col: str):
        spearman = spearmanr(row[col], row[f"pred_{col}"]).statistic
        kendall = kendalltau(row[col], row[f"pred_{col}"]).statistic

        spearman = np.nan_to_num(spearman, nan=0.0)
        kendall = np.nan_to_num(kendall, nan=0.0)

        return {f"{col}_spearman": spearman, f"{col}_kendall": kendall}

    columns = ["cosine", "jaccard", "score"]

    exprs_compute_rank = []
    for col in columns:
        expr = (
            pl.struct(col, f"pred_{col}")
            .map_elements(partial(_map, col=col), return_dtype=pl.Struct)
            .alias(f"{col}_rank")
        )
        exprs_compute_rank.append(expr)
    exprs_unnest = [pl.col(f"{col}_rank").struct.unnest() for col in columns]

    df_rank = df_fold.select(*exprs_compute_rank).select(*exprs_unnest)

    return df_rank
