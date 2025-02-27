import os

import polars as pl
from datasets import Dataset
from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from . import SAVE_PATH

LOG_PATH = SAVE_PATH / "log"


def get_sft_config(
    batch_size: int,
    accumulation_steps: int,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.05,
    epochs: int = 1,
) -> SFTConfig:
    output_dir = (SAVE_PATH / "temp").as_posix()
    return SFTConfig(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )


def get_sft_trainer(
    model,
    config: SFTConfig,
    tokenizer: PreTrainedTokenizerBase,
    collator: DataCollatorForCompletionOnlyLM,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> SFTTrainer:
    return SFTTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )


def save_log(trainer: Trainer, save_name: str) -> pl.DataFrame:
    save_folder = LOG_PATH / save_name
    os.makedirs(save_folder, exist_ok=True)

    log = trainer.state.log_history
    df_log = pl.DataFrame(log)
    df_log.write_json(save_folder / "log.json")
    return df_log
