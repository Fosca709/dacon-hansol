import importlib
from functools import partial
from typing import Literal

import torch.nn as nn
from bitsandbytes.optim import AdamW, PagedAdamW8bit
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers.optimization import get_cosine_schedule_with_warmup

OPTIMIZER_NAMES = {
    "adamw": "get_adamw",
    "paged_adamw_8bit": "get_paged_adamw_8bit",
}

SCHEDULER_NAMES = {
    "cosine": "get_cosine_scheduler",
    "wsd": "get_wsd_scheduler",
}


def get_optimizer(model: nn.Module, optimizer_name: str, lr: float = 6e-5, **optimizer_kwargs) -> Optimizer:
    optim_fn = getattr(importlib.import_module(__name__), OPTIMIZER_NAMES[optimizer_name])
    return optim_fn(model=model, lr=lr, **optimizer_kwargs)


def get_scheduler(
    optimizer: Optimizer, scheduler_name: str, training_steps: int, warmup_ratio: float = 0.0, **scheduler_kwargs
) -> LRScheduler:
    scheduler_fn = getattr(importlib.import_module(__name__), SCHEDULER_NAMES[scheduler_name])
    return scheduler_fn(
        optimizer=optimizer, training_steps=training_steps, warmup_ratio=warmup_ratio, **scheduler_kwargs
    )


def get_training_steps(dataset_size: int, total_batch_size: int) -> int:
    a, b = divmod(dataset_size, total_batch_size)
    if b == 0:
        return a
    else:
        return a + 1


def get_adamw(model: nn.Module, lr: float = 6e-5, betas=(0.9, 0.99), weight_decay=0.1, **kwargs) -> AdamW:
    return AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)


def get_paged_adamw_8bit(
    model: nn.Module, lr: float = 6e-5, betas=(0.9, 0.99), weight_decay=0.1, **kwargs
) -> PagedAdamW8bit:
    return PagedAdamW8bit(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def get_cosine_scheduler(optimizer: Optimizer, training_steps: int, warmup_ratio: float = 0.0, **kwargs) -> LRScheduler:
    warmup_steps = int(training_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=training_steps, **kwargs
    )


def get_wsd_scheduler(
    optimizer: Optimizer,
    training_steps: int,
    warmup_ratio: float = 0.0,
    decay_ratio: float = 0.1,
    num_cycles: int = 1,
    min_lr_rate: float = 0.1,
    decay_fn: Literal["harmonic", "exponential"] = "harmonic",
) -> LRScheduler:
    warmup_steps = int(training_steps * warmup_ratio)
    cycle_steps = int(training_steps / num_cycles)
    decay_steps = int(cycle_steps * decay_ratio)

    lr_lambda = partial(
        _wsd_scheduler,
        cycle_steps=cycle_steps,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        min_lr_rate=min_lr_rate,
        decay_fn=decay_fn,
    )
    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def _wsd_scheduler(
    step: int,
    cycle_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_rate: float,
    decay_fn: Literal["harmonic", "exponential"] = "harmonic",
):
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    remain_steps = cycle_steps - ((step - 1) % cycle_steps)

    if remain_steps > decay_steps:
        return 1

    if decay_fn == "harmonic":
        return wsd_harmonic_decay(remain_steps - 1, decay_steps, min_lr_rate)
    else:
        return wsd_exp_decay(remain_steps - 1, decay_steps, min_lr_rate)


# Exponential decay as used in the original WSD paper: https://arxiv.org/abs/2404.06395
# But the formula in the paper is inaccurate and ambiguous, so this implementation may not exactly match the one in the paper.
def wsd_exp_decay(remain_steps: int, decay_steps: int, min_lr_rate: float):
    min_lr_rate = max(1e-5, min_lr_rate)
    return min_lr_rate ** ((decay_steps - remain_steps) / decay_steps)


# Harmonic decay as used in the WSD-S paper: https://arxiv.org/abs/2410.05192
def wsd_harmonic_decay(remain_steps: int, decay_steps: int, min_lr_rate: float):
    t = decay_steps - remain_steps
    T = decay_steps
    mu = min_lr_rate
    return 1 / ((t / (T * mu)) + 1 - (t / T))
