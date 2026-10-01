# Copyright 2026 Juwon Hwang, Hanyang University
# Licensed under the Apache License, Version 2.0 (see LICENSE-RLCC).

"""AdaRFT-style difficulty sampler + RLCR reward on top of the existing GRPO CustomTrainer.

Nothing in GRPO_Trainer.py / reward_fns.py is modified. This file only adds:
  * rlcr_reward: reward = correctness - (confidence - correctness) ** 2
  * AdaRLCRTrainer: picks, at every optimizer step, the not-yet-used prompts whose fixed difficulty
    is closest to the current target difficulty T, and updates T from the batch correctness rate.
"""

import json
import math
import os
import re
import time
from functools import partial

import numpy as np
import torch
from transformers import TrainerCallback

from GRPO_Trainer import CustomTrainer
from reward_fns import accuracy_reward, format_reward
from trainer_utils import shuffle_tensor_dict, split_tensor_dict

CONF_RE = re.compile(r"<confidence>(.*?)</confidence>", re.DOTALL | re.MULTILINE)

# Filled by rlcr_reward on every call, read by AdaRLCRTrainer right after scoring.
LAST_STATS = {}


def rlcr_reward(format_pattern, completions, answer, invalid_reward=-1.0, **kwargs):
    contents = [c[0]["content"] for c in completions]
    fmt = format_reward(format_pattern, completions)
    correct = accuracy_reward(format_pattern, completions, answer)
    rewards, confs, valid = [], [], []
    for content, fr, cr in zip(contents, fmt, correct):
        q = None
        if fr:
            found = CONF_RE.findall(content)
            try:
                q = float(found[-1])
            except Exception:
                q = None
        if q is None or not (0.0 <= q <= 1.0):
            rewards.append(float(invalid_reward))
            confs.append(float("nan"))
            valid.append(False)
        else:
            rewards.append(float(cr) - (q - float(cr)) ** 2)
            confs.append(q)
            valid.append(True)
    LAST_STATS["correct"] = [float(x) for x in correct]
    LAST_STATS["conf"] = confs
    LAST_STATS["valid"] = valid
    return rewards


def make_rlcr_reward(format_pattern, invalid_reward):
    fn = partial(rlcr_reward, format_pattern, invalid_reward=invalid_reward)
    fn.__name__ = "rlcr_reward"
    return fn


def calibration_stats(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return float("nan"), float("nan")
    brier = float(np.mean((conf - correct) ** 2))
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return brier, float(ece)


class StepTimer(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.t_prev = None

    def on_train_begin(self, args, state, control, **kw):
        self.t_prev = time.time()
        expected = getattr(self.trainer, "expected_steps", None)
        if getattr(self.trainer, "full_run", False) and expected is not None:
            assert state.max_steps == expected, f"HF planned {state.max_steps} optimizer steps, expected {expected}"

    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        self.trainer.step_wall.append(now - self.t_prev)
        self.t_prev = now


class AdaRLCRTrainer(CustomTrainer):
    def __init__(self, *args, ada_cfg, difficulty, **kwargs):
        super().__init__(*args, **kwargs)
        self.ada = ada_cfg
        self.difficulty = np.asarray(difficulty, dtype=float)
        n = len(self.difficulty)
        assert n == len(self.train_dataset)
        self.used = np.zeros(n, dtype=bool)
        self.rng = np.random.default_rng(self.args.seed)
        self.T = float(np.median(self.difficulty)) if ada_cfg["T0"] is None else float(ada_cfg["T0"])
        self.step_wall = []
        self.step_records = []
        self.selection_log = []
        self._loss_scale = 1.0
        self.add_callback(StepTimer(self))
        self.jsonl_path = os.path.join(self.args.output_dir, "adarlcr_steps.jsonl")
        os.makedirs(self.args.output_dir, exist_ok=True)
        open(self.jsonl_path, "w").close()

    # A trivial dataloader: one dummy item per micro-batch. Prompts are chosen adaptively in _prepare_inputs,
    # so the dataloader only has to define how many micro-batches (=> optimizer steps) the epoch has.
    def get_train_dataloader(self):
        from torch.utils.data import DataLoader

        n_micro = math.ceil(len(self.difficulty) * self.num_generations / self.args.per_device_train_batch_size)
        loader = DataLoader(list(range(n_micro)), batch_size=1, collate_fn=lambda x: x, shuffle=False)
        return self.accelerator.prepare(loader)

    def _select(self, k):
        idx = np.flatnonzero(~self.used)
        dist = np.abs(self.difficulty[idx] - self.T)
        tie = self.rng.random(len(idx))
        order = np.lexsort((tie, dist))
        return idx[order[:k]]

    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            return super()._prepare_inputs(generation_batch)
        gpg = self.args.steps_per_generation
        if self._step % gpg == 0 or self._buffered_inputs is None:
            t0 = time.time()
            k = min(self.ada["prompts_per_step"], int((~self.used).sum()))
            chosen = self._select(k)
            self.used[chosen] = True
            T_used = self.T
            rows = [self.train_dataset[int(i)] for i in chosen]
            inputs = [dict(r) for r in rows for _ in range(self.num_generations)]

            batch = self._generate_and_score_completions(inputs)
            t_gen = time.time() - t0

            correct = np.asarray(LAST_STATS["correct"], dtype=float)
            conf = np.asarray(LAST_STATS["conf"], dtype=float)
            valid = np.asarray(LAST_STATS["valid"], dtype=bool)
            corr_rate = float(correct.mean())
            self.T = float(np.clip(T_used + self.ada["eta"] * math.tanh(self.ada["alpha"] * (corr_rate - self.ada["beta"])), 0.0, 100.0))
            brier, ece = calibration_stats(conf[valid], correct[valid])
            rewards = self._metrics["train"]["reward"][-1]
            rec = {
                "step": self.state.global_step + 1,
                "n_prompts": int(k),
                "n_completions": len(inputs),
                "T": T_used,
                "T_next": self.T,
                "mean_difficulty": float(self.difficulty[chosen].mean()),
                "min_difficulty": float(self.difficulty[chosen].min()),
                "max_difficulty": float(self.difficulty[chosen].max()),
                "correctness_rate": corr_rate,
                "mean_confidence": float(conf[valid].mean()) if valid.any() else float("nan"),
                "brier": brier,
                "ece": ece,
                "reward": rewards,
                "format_valid_rate": float(valid.mean()),
                "used_total": int(self.used.sum()),
                "sec_generate_and_score": t_gen,
            }
            self.step_records.append(rec)
            self.selection_log.append({"step": rec["step"], "T": T_used, "indices": [int(i) for i in chosen]})
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            for key, val in rec.items():
                if key not in ("step",):
                    self._metrics["train"][f"adarlcr/{key}"].append(float(val))

            batch = shuffle_tensor_dict(batch)
            n_chunks = len(inputs) // self.args.per_device_train_batch_size
            self._buffered_inputs = split_tensor_dict(batch, n_chunks)
            # HF/accelerate divide every window by gradient_accumulation_steps; the last window has fewer
            # micro-batches, so rescale it to keep a true mean over its micro-batches.
            self._loss_scale = gpg / n_chunks
        inputs = self._buffered_inputs[self._step % gpg]
        self._step += 1
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch) * self._loss_scale

    def save_selection_log(self):
        with open(os.path.join(self.args.output_dir, "adarlcr_selection.json"), "w") as f:
            json.dump(self.selection_log, f)
