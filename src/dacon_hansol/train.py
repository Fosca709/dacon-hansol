import os
from typing import Any, Literal, Optional, Union

import polars as pl
import torch
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


# use `patch_unsloth_grpo` before calling the function
def patch_grpo_trainer_with_ref():
    from contextlib import nullcontext

    import torch.nn.functional as F
    from transformers import Trainer
    from trl.trainer.grpo_trainer import (
        GRPOTrainer,
        apply_chat_template,
        broadcast_object_list,
        gather,
        gather_object,
        is_conversational,
        maybe_apply_chat_template,
        nn,
        pad,
        unwrap_model_for_generation,
    )

    class GRPOwithRef(GRPOTrainer):
        def __init__(
            self,
            model,
            reward_funcs,
            args=None,
            train_dataset=None,
            eval_dataset=None,
            processing_class=None,
            reward_processing_classes=None,
            callbacks=None,
            peft_config=None,
            **kwargs,
        ):
            super().__init__(
                model=model,
                reward_funcs=reward_funcs,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=processing_class,
                reward_processing_classes=reward_processing_classes,
                callbacks=callbacks,
                peft_config=peft_config,
                **kwargs,
            )

        def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
            device = self.accelerator.device
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]

            # Generate completions using either vLLM or regular generation
            if self.args.use_vllm:
                # First, have main process load weights if needed
                if self.state.global_step != self._last_loaded_step:
                    self._move_model_to_vllm()
                    self._last_loaded_step = self.state.global_step

                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(
                        all_prompts_text,
                        sampling_params=self.sampling_params,
                        use_tqdm=False,
                        lora_request=self.model.load_lora("grpo_trainer_lora_model", load_tensors=True),
                    )
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

                # Pad the completions, and concatenate them with the prompts
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            else:
                # Regular generation path
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

                # Compute prompt length and extract completion ids
                prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

            # Replace the last completion with a reference text
            answer = inputs[0]["answer"]
            answer_ids = self.processing_class.encode(answer, add_special_tokens=False, truncation=False)
            answer_ids = answer_ids[-self.max_completion_length + 1 :]
            answer_ids.append(self.processing_class.eos_token_id)

            completion_length = completion_ids.size(-1)
            answer_length = len(answer_ids)

            if answer_length <= completion_length:
                answer_ids.extend(
                    [self.processing_class.pad_token_id for _ in range(completion_length - answer_length)]
                )
            else:
                completion_ids = F.pad(
                    completion_ids, (0, answer_length - completion_length), value=self.processing_class.pad_token_id
                )

            answer_ids = torch.tensor(answer_ids, device=completion_ids.device)
            completion_ids[-1] = answer_ids

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            with (
                torch.inference_mode(),
                torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                    if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                    else torch.bfloat16,
                )
                if not torch.is_autocast_enabled("cuda")
                else nullcontext(),
            ):
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model, keep_fp32_wrapper=False).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )

            # Decode the generated completions
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text

            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with (
                        torch.inference_mode(),
                        torch.amp.autocast(
                            device_type="cuda",
                            dtype=torch.float16
                            if os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16"
                            else torch.bfloat16,
                        )
                        if not torch.is_autocast_enabled("cuda")
                        else nullcontext(),
                    ):
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
            # completions may be distributed across processes
            rewards_per_func = gather(rewards_per_func)

            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

            # Compute grouped-wise rewards
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            # Normalize the rewards to compute the advantages
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

            # Slice to keep only the local part of the data
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            advantages = advantages[process_slice]

            # Log the metrics
            reward_per_func = rewards_per_func.mean(0)
            for i, reward_func in enumerate(self.reward_funcs):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                else:
                    reward_func_name = reward_func.__name__
                self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

            self._metrics["reward"].append(rewards.mean().item())
            self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

            return {
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask,
                "completion_ids": completion_ids,
                "completion_mask": completion_mask,
                "ref_per_token_logps": ref_per_token_logps,
                "advantages": advantages,
            }

    return GRPOwithRef


def get_grpo_trainer(
    model,
    tokenizer,
    config,
    embed_model: SentenceTransformer,
    train_dataset: Dataset,
    print_steps: int = 10,
    check_unsloth: bool = True,
    use_ref_as_sample: bool = False,
):
    from trl import GRPOConfig, GRPOTrainer

    assert isinstance(config, GRPOConfig)
    if check_unsloth:
        assert "Unsloth" in GRPOTrainer.__name__

    def dampening_rewards(rewards, t=0.7):
        r = max(rewards[:-1])
        rewards[-1] = t * r + (1 - t) * 1.0
        return rewards

    def jaccard_reward(completions, answer, **kwargs):
        comp_texts = [c[0]["content"] for c in completions]
        rewards = [jaccard_similarity(t1, t2) for t1, t2 in zip(answer, comp_texts)]

        if use_ref_as_sample:
            rewards = dampening_rewards(rewards)

        return rewards

    def cosine_reward(completions, answer, **kwargs):
        comp_texts = [c[0]["content"] for c in completions]

        comp_embed = embed_model.encode(comp_texts, show_progress_bar=False)
        ans_embed = embed_model.encode(answer, show_progress_bar=False)

        rewards = cosine_similarity(ans_embed, comp_embed).clip(min=0).tolist()

        if use_ref_as_sample:
            rewards = dampening_rewards(rewards)

        return rewards

    if use_ref_as_sample:
        if config.per_device_train_batch_size != config.num_generations:
            raise Exception("`batch_size` must be the same as `num_generations` for this trainer.")
        trainer_class = patch_grpo_trainer_with_ref()
    else:
        trainer_class = GRPOTrainer

    trainer = trainer_class(
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
    use_ref_as_sample: bool = False,
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
        use_ref_as_sample=use_ref_as_sample,
    )

    trainer.train()
    save_log(trainer, save_name)

    if save_model:
        save_path = SAVE_PATH / "model" / save_name
        save_peft_model(model, tokenizer, save_path)

        if push_to_hub:
            hf_upload_folder(save_path)
