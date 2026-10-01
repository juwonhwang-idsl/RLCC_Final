"""RLCR reward + AdaRFT difficulty sampler on the existing GRPO trainer (Qwen3-1.7B).

Run through scripts/run_adarlcr.sh. Defaults follow the request: 5,000 prompts, 1 epoch,
64 prompts per optimizer step, G=32, T0=median difficulty, beta=0.5, alpha=2, eta=10.
"""

import json
import math
import os

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import set_seed
from trl import TrlParser, get_peft_config

from adarlcr_trainer import AdaRLCRTrainer, make_rlcr_reward
from arguments import GRPOConfig, GRPOScriptArguments, ModelConfig
from dataset_processing import process_dataset
from rl_runner import logger, logger_setup, model_init


def env_float(name, default):
    return float(os.environ.get(name, default))


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args)

    ada_cfg = {
        "prompts_per_step": int(os.environ.get("ADARLCR_PROMPTS", 64)),
        "eta": env_float("ADARLCR_ETA", 10),
        "alpha": env_float("ADARLCR_ALPHA", 2),
        "beta": env_float("ADARLCR_BETA", 0.5),
        "T0": None,
        "invalid_reward": env_float("ADARLCR_INVALID_REWARD", -1.0),
    }
    G = training_args.num_generations
    n_prompts = ada_cfg["prompts_per_step"]
    assert training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps == n_prompts * G, (
        "micro-batch x accumulation must equal prompts_per_step x num_generations"
    )
    assert training_args.generation_batch_size == n_prompts * G, training_args.generation_batch_size

    raw = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_raw = raw[script_args.dataset_train_split].select(range(script_args.train_subset_size))
    solve_rate = np.asarray(train_raw["llama8b_solve_rate"], dtype=float)
    difficulty = 100.0 * (1.0 - solve_rate)
    assert len(difficulty) == script_args.train_subset_size and 0.0 <= difficulty.min() and difficulty.max() <= 100.0

    dataset = process_dataset(DatasetDict({script_args.dataset_train_split: train_raw}), script_args)
    train_dataset = dataset[script_args.dataset_train_split]
    if "messages" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("messages")
    assert len(train_dataset) == len(difficulty)

    expected_steps = math.ceil(len(train_dataset) / n_prompts)
    logger.info("AdaRLCR config: %s | difficulty median=%.3f (T0) | expected optimizer steps=%d",
                ada_cfg, float(np.median(difficulty)), expected_steps)
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "adarlcr_config.json"), "w") as f:
        json.dump({**ada_cfg, "T0_resolved": float(np.median(difficulty)), "expected_steps": expected_steps,
                   "num_generations": G, "difficulty": "100*(1-llama8b_solve_rate)"}, f, indent=2)

    training_args.model_init_kwargs = model_init(model_args, training_args)
    full_run = training_args.max_steps <= 0
    if full_run:
        # HF floors the epoch's last partial accumulation window; pin the step count so the final
        # (smaller) batch is trained on and the LR schedule spans exactly `expected_steps`.
        training_args.max_steps = expected_steps
    trainer = AdaRLCRTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[make_rlcr_reward(script_args.format_pattern, ada_cfg["invalid_reward"])],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        ada_cfg=ada_cfg,
        difficulty=difficulty,
    )

    trainer.full_run = full_run
    trainer.expected_steps = expected_steps
    if full_run:
        n_micro = len(trainer.get_train_dataloader())
        planned = n_micro // training_args.gradient_accumulation_steps + int(n_micro % training_args.gradient_accumulation_steps > 0)
        assert planned == expected_steps, f"planned optimizer steps {planned} != {expected_steps}"
        if script_args.train_subset_size == 5000 and n_prompts == 64:
            assert expected_steps == 79, f"expected {expected_steps} optimizer steps, not 79"
        logger.info("Planned optimizer steps: %d (last window = %d micro-batches)", planned,
                    n_micro % training_args.gradient_accumulation_steps or training_args.gradient_accumulation_steps)

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_selection_log()

    steps_done = trainer.state.global_step
    if full_run:
        assert steps_done == expected_steps, f"finished {steps_done} optimizer steps, expected {expected_steps}"
        if script_args.train_subset_size == 5000 and n_prompts == 64:
            assert steps_done == 79
        assert bool(trainer.used.all()) and int(trainer.used.sum()) == len(train_dataset), "not every prompt used exactly once"
        used = [i for rec in trainer.selection_log for i in rec["indices"]]
        assert len(used) == len(set(used)) == len(train_dataset), "duplicate or missing prompts"
    else:
        assert steps_done == training_args.max_steps

    summary = {
        "steps": steps_done,
        "train_runtime": train_result.metrics.get("train_runtime"),
        "step_wall_seconds": trainer.step_wall,
        "mean_step_seconds": float(np.mean(trainer.step_wall)) if trainer.step_wall else None,
    }
    with open(os.path.join(training_args.output_dir, "adarlcr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary: %s", {k: v for k, v in summary.items() if k != "step_wall_seconds"})

    if os.environ.get("SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}:
        return
    trainer.save_model(training_args.output_dir)
    logger.info("Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
