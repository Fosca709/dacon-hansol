import os

import polars as pl
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from transformers import Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from trl import DataCollatorForCompletionOnlyLM, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from . import SAVE_PATH
from .data import get_grpo_dataset, get_llama_collator, get_trl_dataset, train_val_split
from .inference import cosine_similarity, jaccard_similarity
from .model import save_peft_model
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
    batch_size: int = 1,
    accumulation_steps: int = 1,
    num_generations: int = 8,
    max_prompt_length: int = 1024,
    max_completion_length: int = 128,
    temperature: float = 0.9,
    learning_rate: float = 5e-6,
    use_vllm: bool = False,
    max_steps: int = -1,
    optim: str = "adamw_torch",
    eval_strategy: str = "epoch",
) -> GRPOConfig:
    output_dir = (SAVE_PATH / "temp").as_posix()
    return GRPOConfig(
        max_prompt_length=max_prompt_length,
        num_generations=num_generations,
        temperature=temperature,
        max_completion_length=max_completion_length,
        use_vllm=use_vllm,
        learning_rate=learning_rate,
        reward_weights=[0.7, 0.3],
        output_dir=output_dir,
        eval_strategy=eval_strategy,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.1,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.99,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim=optim,
    )


def get_grpo_trainer(
    model,
    tokenizer,
    embed_model: SentenceTransformer,
    config: GRPOConfig,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> GRPOTrainer:
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

    return GRPOTrainer(
        model=model,
        reward_funcs=[cosine_reward, jaccard_reward],
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )


# use `patch_unsloth_grpo` before calling the function and the model
def grpo_train(
    model,
    tokenizer,
    embed_model: SentenceTransformer,
    run_name: str,
    batch_size: int = 1,
    accumulation_stpes: int = 1,
    num_generations: int = 8,
    learning_rate: float = 6e-5,
    use_vllm: bool = False,
    max_steps: int = -1,
    save_model: bool = True,
    push_to_hub: bool = True,
    debug_mode: bool = True,
    do_val: bool = False,
    optim: str = "adamw_torch",
) -> None:
    save_name = get_save_name(run_name)

    df_train, df_val = train_val_split()
    if debug_mode:
        df_train = df_train[:20]
        df_val = df_val[:20]
    train_dataset = get_grpo_dataset(df_train)
    eval_dataset = get_grpo_dataset(df_val) if do_val else None
    eval_strategy = "epoch" if do_val else "no"

    config = get_grpo_config(
        batch_size=batch_size,
        accumulation_steps=accumulation_stpes,
        num_generations=num_generations,
        learning_rate=learning_rate,
        use_vllm=use_vllm,
        max_steps=max_steps,
        optim=optim,
        eval_strategy=eval_strategy,
    )

    trainer = get_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        embed_model=embed_model,
        config=config,
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
