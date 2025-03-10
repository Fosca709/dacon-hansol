import os
from typing import Literal, Optional

import polars as pl
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils.notebook import NotebookProgressCallback, NotebookTrainingTracker
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from . import SAVE_PATH
from .data import get_grpo_dataset, get_llama_collator, get_trl_dataset, train_val_split
from .inference import cosine_similarity, jaccard_similarity
from .model import save_peft_model
from .optimizer import get_training_steps
from .utils import get_save_name, hf_upload_folder

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


def sft_train(
    model,
    tokenizer,
    run_name: str,
    batch_size: int = 1,
    accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    debug_mode: bool = False,
    save_model: bool = True,
    push_to_hub: bool = True,
) -> None:
    save_name = get_save_name(run_name)

    config = get_sft_config(batch_size=batch_size, accumulation_steps=accumulation_steps, learning_rate=learning_rate)

    df_train, df_val = train_val_split()
    if debug_mode:
        df_train = df_train[:50]
        df_val = df_val[:50]
    train_dataset = get_trl_dataset(df_train, tokenizer)
    eval_dataset = get_trl_dataset(df_val, tokenizer)

    trainer = get_sft_trainer(
        model=model,
        config=config,
        tokenizer=tokenizer,
        collator=get_llama_collator(tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    save_log(trainer, save_name)

    if save_model:
        save_path = SAVE_PATH / "model" / save_name
        save_peft_model(model, tokenizer, save_path)

        if push_to_hub:
            hf_upload_folder(save_path)


def patch_unsloth_grpo() -> None:
    from unsloth import FastLanguageModel, PatchFastRL

    PatchFastRL("GRPO", FastLanguageModel)


def get_grpo_config(
    batch_size: int = 2,
    accumulation_steps: int = 1,
    num_generations: int = 2,
    max_prompt_length: int = 1024,
    max_completion_length: int = 128,
    temperature: float = 0.9,
    max_grad_norm: float = 0.2,
    learning_rate: float = 3e-6,
    optimizer: Literal["adamw_torch", "paged_adamw_8bit"] = "paged_adamw_8bit",
    scheduler: Literal["cosine", "wsd"] = "wsd",
    warmup_ratio: float = 0.1,
    beta: float = 0.001,
    use_vllm: bool = False,
    max_steps: int = -1,
    training_steps: int = -1,
    seed: int = 42,
    check_unsloth: bool = True,
):
    from trl import GRPOConfig

    scheduler_kwargs = {}
    if scheduler == "wsd":
        if max_steps == -1 and training_steps == -1:
            raise Exception("To use the WSD scheduler, you must specify either max_steps or training_steps.")

        decay_ratio = 0.1
        if max_steps != -1:
            decay_steps = int(max_steps * decay_ratio)

        else:
            decay_steps = int(training_steps * decay_ratio)

        scheduler = "warmup_stable_decay"
        scheduler_kwargs = {"num_decay_steps": decay_steps, "min_lr_ratio": 0.1}

    if check_unsloth:
        assert "Unsloth" in GRPOConfig.__name__
    output_dir = (SAVE_PATH / "temp").as_posix()
    return GRPOConfig(
        max_prompt_length=max_prompt_length,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        reward_weights=[0.7, 0.3],
        output_dir=output_dir,
        eval_strategy="no",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        beta=beta,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        seed=seed,
        data_seed=seed,
        adam_beta2=0.99,
        learning_rate=learning_rate,
        optim=optimizer,
        lr_scheduler_type=scheduler,
        lr_scheduler_kwargs=scheduler_kwargs,
        warmup_ratio=warmup_ratio,
    )


class GRPONotebookCallback(NotebookProgressCallback):
    def __init__(self, print_steps: int = 10):
        super().__init__()
        self.print_steps = print_steps

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_loss = 0
        self.last_log = 0
        column_names = [
            "Step",
            "Loss",
            "Reward",
            "Reward std",
            "Cosine",
            "Jaccard",
            "KL",
            "Grad Norm",
            "Completion Length",
        ]
        self.training_tracker = NotebookTrainingTracker(state.max_steps, column_names)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (state.global_step % self.print_steps == 0) and ("loss" in logs):
            values = {
                "Step": state.global_step,
                "Loss": logs["loss"],
                "Reward": logs["reward"],
                "Reward std": logs["reward_std"],
                "Cosine": logs["rewards/cosine_reward"],
                "Jaccard": logs["rewards/jaccard_reward"],
                "KL": logs["kl"],
                "Grad Norm": logs["grad_norm"],
                "Completion Length": logs["completion_length"],
            }

            self.training_tracker.inner_table.append(values.values())


def get_grpo_trainer(
    model,
    tokenizer,
    config,
    embed_model: SentenceTransformer,
    train_dataset: Dataset,
    print_steps: int = 10,
    check_unsloth: bool = True,
):
    from trl import GRPOConfig, GRPOTrainer

    assert isinstance(config, GRPOConfig)
    if check_unsloth:
        assert "Unsloth" in GRPOTrainer.__name__

    def jaccard_reward(completions, answer, **kwargs):
        comp_texts = [c[0]["content"] for c in completions]
        rewards = [jaccard_similarity(t1, t2) for t1, t2 in zip(answer, comp_texts)]
        return rewards

    def cosine_reward(completions, answer, **kwargs):
        comp_texts = [c[0]["content"] for c in completions]

        comp_embed = embed_model.encode(comp_texts, show_progress_bar=False)
        ans_embed = embed_model.encode(answer, show_progress_bar=False)

        rewards = cosine_similarity(ans_embed, comp_embed).clip(min=0).tolist()
        return rewards

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[cosine_reward, jaccard_reward],
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[GRPONotebookCallback(print_steps=print_steps)],
    )

    for callback in trainer.callback_handler.callbacks:
        if callback.__class__.__name__ == "NotebookProgressCallback":
            trainer.remove_callback(callback)

    return trainer


# use `patch_unsloth_grpo` before calling the function and the model
def grpo_train(
    model,
    tokenizer,
    embed_model: SentenceTransformer,
    run_name: str,
    df_train: Optional[pl.DataFrame] = None,
    batch_size: int = 2,
    accumulation_steps: int = 1,
    num_generations: int = 2,
    max_steps: int = -1,
    learning_rate: float = 3e-6,
    optimizer: Literal["adamw_torch", "paged_adamw_8bit"] = "paged_adamw_8bit",
    scheduler: Literal["cosine", "wsd"] = "wsd",
    warmup_ratio: float = 0.1,
    temperature: float = 0.9,
    max_grad_norm: float = 0.2,
    beta: float = 0.001,
    use_vllm: bool = False,
    save_model: bool = True,
    push_to_hub: bool = True,
    print_steps: int = 10,
    seed: int = 42,
    check_unsloth: bool = True,
) -> None:
    save_name = get_save_name(run_name)

    if df_train is None:
        df_train, _ = train_val_split()
    train_dataset = get_grpo_dataset(df_train)

    if max_steps == -1:
        total_batch_size = int(batch_size / num_generations) * accumulation_steps
        training_steps = get_training_steps(len(train_dataset), total_batch_size)
    else:
        training_steps = max_steps

    config = get_grpo_config(
        batch_size=batch_size,
        accumulation_steps=accumulation_steps,
        num_generations=num_generations,
        temperature=temperature,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup_ratio=warmup_ratio,
        beta=beta,
        use_vllm=use_vllm,
        max_steps=max_steps,
        training_steps=training_steps,
        seed=seed,
        check_unsloth=check_unsloth,
    )

    trainer = get_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        embed_model=embed_model,
        train_dataset=train_dataset,
        print_steps=print_steps,
        check_unsloth=check_unsloth,
    )

    trainer.train()
    save_log(trainer, save_name)

    if save_model:
        save_path = SAVE_PATH / "model" / save_name
        save_peft_model(model, tokenizer, save_path)

        if push_to_hub:
            hf_upload_folder(save_path)
