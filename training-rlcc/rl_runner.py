import os
import sys
import faulthandler
TRL_LOCAL_ROOT = os.path.join(os.path.dirname(__file__), "trl")
if TRL_LOCAL_ROOT not in sys.path:
    sys.path.insert(0, TRL_LOCAL_ROOT)
from arguments import GRPOScriptArguments,GRPOConfig,ModelConfig
from trl import TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config
from transformers import set_seed, AutoTokenizer
import logging
import transformers
import datasets 
from datasets import DatasetDict, load_dataset, load_from_disk
import gc
import json
import re
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftConfig, PeftModel
from reward_fns import (
    format_reward,
    accuracy_reward,
    abstention_accuracy_reward,
    brier_reward,
    mean_confidence_reward,
    confidence_one_or_zero,
)
from system_prompts import get_sys_prompt
from dataset_processing import process_dataset 
from GRPO_Trainer import CustomTrainer
import torch
from functools import partial


logger = logging.getLogger(__name__)
CONFIDENCE_PATTERN = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL | re.MULTILINE)
faulthandler.enable()


def logger_setup(script_args, training_args, model_args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

def model_init(model_args, training_args):
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["low_cpu_mem_usage"] = True
    return model_kwargs


def extract_last_confidence(text):
    matches = CONFIDENCE_PATTERN.findall(text or "")
    if not matches:
        return 0.0
    try:
        confidence = float(matches[-1].strip())
    except Exception:
        return 0.0
    return max(0.0, min(confidence, 1.0))




def build_reward_funcs(script_args):
    registry = {
        "format": partial(format_reward, format_pattern=script_args.format_pattern),
        "accuracy": partial(accuracy_reward, format_pattern=script_args.format_pattern),
        "abstention_accuracy": partial(
            abstention_accuracy_reward,
            format_pattern=script_args.format_pattern,
            abstain_reward=script_args.abstain_reward_lambda,
        ),
        "brier": partial(brier_reward, format_pattern=script_args.format_pattern),
        "mean_confidence": mean_confidence_reward,
        "confidence_one_or_zero": confidence_one_or_zero,
    }
    return [registry[func] for func in script_args.reward_funcs]



def main(script_args, training_args, model_args):
    set_seed(training_args.seed)
    logger_setup(script_args, training_args, model_args) 
    skip_final_save = os.environ.get("SKIP_FINAL_SAVE", "").lower() in {"1", "true", "yes"}

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    reward_funcs = build_reward_funcs(script_args)

    dataset = process_dataset(dataset, script_args)  

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    if script_args.rlcc_preordered_dataset_path:
        logger.info(
            "*** RLCC: loading preordered train dataset from %s ***",
            script_args.rlcc_preordered_dataset_path,
        )
        preordered_train = load_from_disk(script_args.rlcc_preordered_dataset_path)
        preordered_dataset = DatasetDict({script_args.dataset_train_split: preordered_train})
        preordered_dataset = process_dataset(preordered_dataset, script_args)
        if "messages" in preordered_dataset[script_args.dataset_train_split].column_names:
            preordered_dataset[script_args.dataset_train_split] = preordered_dataset[
                script_args.dataset_train_split
            ].remove_columns("messages")
        train_dataset = preordered_dataset[script_args.dataset_train_split]
    if script_args.train_subset_size is not None:
        train_dataset = train_dataset.select(range(script_args.train_subset_size))
    if script_args.eval_subset_size is not None:
        eval_dataset = eval_dataset.select(range(script_args.eval_subset_size))
        
    model_init_kwargs = model_init(model_args, training_args)
    training_args.model_init_kwargs = model_init_kwargs

    trainer = CustomTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = script_args.train_subset_size
    try:
        trainer.save_state()
    except Exception:
        print("Failed to save state, please debug")
        pass

    if skip_final_save:
        logger.info("*** Skipping final model save because SKIP_FINAL_SAVE is enabled ***")
        return

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["rl-verify"],
    }
    if trainer.accelerator.is_main_process:
        try:
            trainer.create_model_card(**kwargs)
        except Exception as exc:
            logger.warning("Skipping model card creation after save: %s", exc)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

    
