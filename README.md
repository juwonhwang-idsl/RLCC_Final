# RLCC: Reinforcement Learning with Confidence Curriculum

[English](#english) | [한국어](#한국어)

---

## English

Code and configs for **"Confidence as Curriculum: Reinforcement Learning for Joint Reasoning and Calibration"** (under review).

Reinforcement Learning with Verifiable Rewards (RLVR) improves reasoning but leaves models overconfident. Reinforcement Learning with Calibration Rewards (RLCR) fixes this with a Brier-score reward, but trades away reasoning accuracy, especially in smaller models. **RLCC** treats the calibrated confidence produced by an RLCR checkpoint as an intrinsic difficulty signal, reorders the training data from easy to hard, and restarts this progression across *K* curriculum segments to mitigate forgetting. The reward and learning algorithm are unchanged from RLCR -- only the order in which data is presented changes. Across mathematical reasoning and multi-hop QA, and across model scales (0.6B-4B) and architectures (Qwen3, Phi-4-mini, Llama-3.2), RLCC recovers most of the reasoning accuracy lost to RLCR while maintaining or improving calibration.

### Repository structure

```
training-rl/      RLVR / RLCR training code (no curriculum support)
training-rlcc/     RLCC training code: curriculum loading, K-segment restarts,
                   difficulty-signal variants (solve-rate, answer-NLL), and
                   AdaRFT+RLCR (R05)
evaluation/        Evaluation harness (accuracy, ECE, PCE, Brier, AUROC)
scripts/           Curriculum construction (build_rlcc_dataset.py, ...) and
                   launch wrappers
configs/
  rl/              RLVR/RLCR training configs -> run with training-rl/
  rlcc/            RLCC training configs -> run with training-rlcc/
  eval_configs/     Evaluation configs for every result in the paper
results/           Main result numbers (accuracy/ECE/PCE/Brier/AUROC). No raw generations,
                   just the paper's table numbers. E/A/D/K entries are transcribed directly
                   from the paper's tables; R01/R03/R04/R06/R07 are from local eval runs.
                   R05 (AdaRFT+RLCR) ran only on a separate A100 server and isn't included yet.
```

`configs/rl/` and `configs/rlcc/` are run with different codebases on purpose:
`training-rl/` has no curriculum-loading support, while `training-rlcc/` does.
Always match the config directory to the matching training directory
(`run_training.sh` does this for you).

### Config naming

Configs are named after the paper result they produce:

| Prefix | Paper content |
|---|---|
| `E01`-`E18` | Main results (Tables 1-4): RLVR / RLCR / RLCC across 5 models x 2 domains |
| `A01`-`A05` | RLCC-Ascending (RLCC-A) ablation, one per model (Table 5, Appendix E) |
| `D01`-`D02` | Difficulty-signal ablation: solve-rate / answer-NLL vs. confidence (Table 6) |
| `K01`-`K04` | Restart-count sweep, K in {1,2,4,5} (K=3 is E03) (Table 7) |
| `R01`-`R07` | Experiments added during rebuttal (to be incorporated into the paper later) |

### Datasets

All configs reference datasets by their public Hugging Face path, no local files needed:

| Dataset | Source |
|---|---|
| Big-Math-digits (train) | [`mehuldamani/big-math-digits`](https://huggingface.co/datasets/mehuldamani/big-math-digits) |
| HotpotQA-Modified (train) | [`mehuldamani/hotpot_qa`](https://huggingface.co/datasets/mehuldamani/hotpot_qa) |
| Big-Math-digits held-out eval | [`juwon1105/hotpot_qa_train_heldout1000`](https://huggingface.co/datasets/juwon1105/hotpot_qa_train_heldout1000) (Hotpot) / `mehuldamani/big-math-digits` test split (Big-Math) |
| MATH500 | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| AIME 2024 | [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| AIME 2025 | [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) |
| AMC23 | [`knoveleng/AMC-23`](https://huggingface.co/datasets/knoveleng/AMC-23) |
| AMC24 | [`juwon1105/amc2024`](https://huggingface.co/datasets/juwon1105/amc2024) |
| 2WikiMultiHopQA (OOD) | [`juwon1105/2wikimultihopqa_val500`](https://huggingface.co/datasets/juwon1105/2wikimultihopqa_val500) |
| MuSiQue (OOD) | [`juwon1105/musique_eval_baseline500`](https://huggingface.co/datasets/juwon1105/musique_eval_baseline500) |
| WikiHop (OOD) | [`juwon1105/wikihop_val500`](https://huggingface.co/datasets/juwon1105/wikihop_val500) |

Model checkpoints referenced by `configs/eval_configs/` are likewise published under [`juwon1105`](https://huggingface.co/juwon1105) on Hugging Face.

### Setup

```bash
conda env create -f environment.yml
pip install -r requirements-rl-qwen3.txt
```

### Running

```bash
# Training: pick the codebase matching the config directory
bash run_training.sh configs/rl/E02-rlcr-1.7B-bigmath.yaml training-rl 0
bash run_training.sh configs/rlcc/E03-rlcc-k3-1.7B-bigmath.yaml training-rlcc 0

# Evaluation
bash run_eval.sh configs/eval_configs/E01-E02-E03-A01-5models-1.7B-bigmath-heldout.json 0
```

### License

This repository mixes two licenses at the file level:

- **`LICENSE` (MIT)**: code carried over from, or modified from, [RLCR](https://github.com/damanimehul/RLCR) (Damani et al., 2026). This covers `training-rl/`, `evaluation/`, and also `GRPO_Trainer.py`/`reward_fns.py`/`rl_runner.py`/`arguments.py`/`trainer_utils.py`/`system_prompts.py`/`dataset_processing.py` inside `training-rlcc/` -- these add curriculum loading on top of RLCR's code but are still derivatives of it.
- **`LICENSE-RLCC` (ISC License, Copyright Juwon Hwang)**: files written entirely from scratch, with nothing carried over from RLCR -- `training-rlcc/adarlcr_trainer.py`, `training-rlcc/rl_runner_adarlcr.py`, `scripts/build_rlcc_dataset.py`, `scripts/build_rlcc_single_sweep.py`, `scripts/build_rlcc_block_shuffle_control.py`, `run_training.sh`, `run_eval.sh`, `scripts/run_adarlcr.sh`. Each of these files carries its own license header at the top.

Config files (`configs/`) are treated as MIT.

### Citation

This work is currently under anonymous review. A citation (and arXiv link) will be added here after the review process concludes.

---

## 한국어

**"Confidence as Curriculum: Reinforcement Learning for Joint Reasoning and Calibration"** (심사 중) 논문의 코드와 config입니다.

Reinforcement Learning with Verifiable Rewards (RLVR)는 추론 능력을 향상시키지만 모델을 과신(overconfident)하게 만듭니다. Reinforcement Learning with Calibration Rewards (RLCR)는 Brier score 기반 보상으로 이를 완화하지만, 그 대가로 추론 정확도를 잃습니다 (특히 작은 모델에서 두드러짐). **RLCC**는 RLCR 체크포인트가 산출한 calibrated confidence를 내재적 난이도 신호로 사용해 학습 데이터를 쉬운 순서에서 어려운 순서로 재배열하고, 망각을 완화하기 위해 이 easy-to-hard 진행을 *K*개의 curriculum segment로 나누어 반복(restart)합니다. 보상 함수와 학습 알고리즘은 RLCR과 동일하며, 오직 데이터가 제시되는 순서만 다릅니다. 수학 추론과 multi-hop QA, 그리고 여러 모델 규모(0.6B-4B)와 아키텍처(Qwen3, Phi-4-mini, Llama-3.2)에 걸쳐, RLCC는 RLCR이 희생한 추론 정확도 대부분을 회복하면서 calibration은 유지하거나 개선합니다.

### 저장소 구조

```
training-rl/      RLVR / RLCR 학습 코드 (curriculum 로딩 미지원)
training-rlcc/     RLCC 학습 코드: curriculum 로딩, K-segment restart,
                   difficulty-signal variant (solve-rate, answer-NLL),
                   AdaRFT+RLCR (R05)
evaluation/        평가 하네스 (accuracy, ECE, PCE, Brier, AUROC)
scripts/           Curriculum 구성 스크립트(build_rlcc_dataset.py 등)와
                   실행 wrapper
configs/
  rl/              RLVR/RLCR 학습 config -> training-rl/ 로 실행
  rlcc/            RLCC 학습 config -> training-rlcc/ 로 실행
  eval_configs/     논문의 모든 결과에 대응하는 평가 config
results/           주요 결과 수치 (accuracy/ECE/PCE/Brier/AUROC). 모델이 생성한 원문 텍스트는
                   포함하지 않고, 논문 표 숫자만 담았습니다. E/A/D/K 항목은 논문 Table에서 그대로
                   옮겼고, R01/R03/R04/R06/R07은 실제로 실행된 로컬 결과입니다. R05(AdaRFT+RLCR)는
                   A100 서버에서만 실행되어 아직 포함되지 않았습니다.
```

`configs/rl/`과 `configs/rlcc/`는 의도적으로 서로 다른 코드베이스로 실행하도록 분리되어 있습니다. `training-rl/`은 curriculum 로딩 기능이 없고, `training-rlcc/`는 있습니다. 항상 config 디렉토리와 짝이 맞는 학습 디렉토리를 사용하세요 (`run_training.sh`가 이를 자동으로 맞춰줍니다).

### Config 이름 규칙

각 config는 논문의 어느 결과에 대응하는지를 이름에 그대로 반영합니다:

| 접두사 | 논문 내용 |
|---|---|
| `E01`-`E18` | 메인 결과 (Table 1-4): 5개 모델 x 2개 도메인에 대한 RLVR / RLCR / RLCC |
| `A01`-`A05` | RLCC-Ascending (RLCC-A) ablation, 모델별 1개 (Table 5, Appendix E) |
| `D01`-`D02` | Difficulty-signal ablation: solve-rate / answer-NLL vs. confidence (Table 6) |
| `K01`-`K04` | Restart 횟수 스윕, K ∈ {1,2,4,5} (K=3는 E03) (Table 7) |
| `R01`-`R07` | Rebuttal에서 추가된 실험 (추후 논문에 반영 예정) |

### 데이터셋

모든 config는 로컬 파일 없이 공개 Hugging Face 경로로 데이터셋을 참조합니다:

| 데이터셋 | 출처 |
|---|---|
| Big-Math-digits (학습용) | [`mehuldamani/big-math-digits`](https://huggingface.co/datasets/mehuldamani/big-math-digits) |
| HotpotQA-Modified (학습용) | [`mehuldamani/hotpot_qa`](https://huggingface.co/datasets/mehuldamani/hotpot_qa) |
| held-out 평가셋 | [`juwon1105/hotpot_qa_train_heldout1000`](https://huggingface.co/datasets/juwon1105/hotpot_qa_train_heldout1000) (Hotpot) / `mehuldamani/big-math-digits`의 test split (Big-Math) |
| MATH500 | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) |
| AIME 2024 | [`HuggingFaceH4/aime_2024`](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) |
| AIME 2025 | [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025) |
| AMC23 | [`knoveleng/AMC-23`](https://huggingface.co/datasets/knoveleng/AMC-23) |
| AMC24 | [`juwon1105/amc2024`](https://huggingface.co/datasets/juwon1105/amc2024) |
| 2WikiMultiHopQA (OOD) | [`juwon1105/2wikimultihopqa_val500`](https://huggingface.co/datasets/juwon1105/2wikimultihopqa_val500) |
| MuSiQue (OOD) | [`juwon1105/musique_eval_baseline500`](https://huggingface.co/datasets/juwon1105/musique_eval_baseline500) |
| WikiHop (OOD) | [`juwon1105/wikihop_val500`](https://huggingface.co/datasets/juwon1105/wikihop_val500) |

`configs/eval_configs/`가 참조하는 모델 체크포인트들도 마찬가지로 Hugging Face [`juwon1105`](https://huggingface.co/juwon1105) 계정에 공개되어 있습니다.

### 설치

```bash
conda env create -f environment.yml
pip install -r requirements-rl-qwen3.txt
```

### 실행

```bash
# 학습: config 디렉토리에 맞는 코드베이스를 선택
bash run_training.sh configs/rl/E02-rlcr-1.7B-bigmath.yaml training-rl 0
bash run_training.sh configs/rlcc/E03-rlcc-k3-1.7B-bigmath.yaml training-rlcc 0

# 평가
bash run_eval.sh configs/eval_configs/E01-E02-E03-A01-5models-1.7B-bigmath-heldout.json 0
```

### 라이센스

이 저장소는 파일 단위로 두 라이센스가 섞여 있습니다:

- **`LICENSE` (MIT)**: [RLCR](https://github.com/damanimehul/RLCR) (Damani et al., 2026)에서 그대로 가져왔거나 그걸 기반으로 수정한 코드. `training-rl/`, `evaluation/`, 그리고 `training-rlcc/` 안의 `GRPO_Trainer.py`/`reward_fns.py`/`rl_runner.py`/`arguments.py`/`trainer_utils.py`/`system_prompts.py`/`dataset_processing.py`도 RLCR 파생물이라 여기 포함됩니다 (curriculum 로딩 등 기능을 추가했지만 원본 구조를 그대로 이어받았습니다).
- **`LICENSE-RLCC` (ISC License, Copyright Juwon Hwang)**: RLCR 코드를 전혀 가져오지 않고 완전히 새로 작성한 파일만 — `training-rlcc/adarlcr_trainer.py`, `training-rlcc/rl_runner_adarlcr.py`, `scripts/build_rlcc_dataset.py`, `scripts/build_rlcc_single_sweep.py`, `scripts/build_rlcc_block_shuffle_control.py`, `run_training.sh`, `run_eval.sh`, `scripts/run_adarlcr.sh`. 각 파일 맨 위에 라이센스 표시가 있습니다.

Config(`configs/`) 파일들은 모두 MIT 쪽으로 취급합니다.

### 인용

현재 익명 심사가 진행 중인 논문입니다. 심사 종료 후 인용 정보(및 arXiv 링크)를 이곳에 추가할 예정입니다.
