# OMTG

Official training code for **Towards One-to-Many Temporal Grounding**.

This repository contains the released training pipeline for:

- supervised fine-tuning (SFT)
- reinforcement learning (RL / GRPO)

## Installation

### SFT

```bash
bash sft/install_qwen3vl.sh
```

### RL

```bash
bash rl/scripts/install_rl.sh
```

## Data Preparation

### SFT

Place the SFT data under `data/`. The default config uses the JSONL files listed in `sft/qwen3vl_4b_omtg_wcot.yaml`.

### RL

Prepare one training parquet file and one evaluation parquet file, then pass them through environment variables when launching training.

## Training

### SFT

```bash
bash sft/train_qwen3vl.sh sft/qwen3vl_4b_omtg_wcot.yaml
```

### RL

Activate the RL environment first:

```bash
source rl/.venv/bin/activate
```

Set the SFT checkpoint and parquet files first:

```bash
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PARQUET=/path/to/train.parquet \
EVAL_PARQUET=/path/to/eval.parquet \
bash rl/scripts/grpo_4b_coldstart_caption_agent.sh
```

To run RL without caption reward:

```bash
TG_REWARD_STRATEGY=tiouformatf1cacc \
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PARQUET=/path/to/train.parquet \
EVAL_PARQUET=/path/to/eval.parquet \
bash rl/scripts/grpo_4b_coldstart_caption_agent.sh
```

To run RL with caption reward:

```bash
OMTG_JUDGE_MODEL=your_judge_model \
OMTG_JUDGE_API_KEY=your_api_key \
MODEL_PATH=/path/to/sft_checkpoint \
TRAIN_PARQUET=/path/to/train.parquet \
EVAL_PARQUET=/path/to/eval.parquet \
bash rl/scripts/grpo_4b_coldstart_caption_agent.sh
```

If needed, `OMTG_JUDGE_BASE_URL` can be set for a custom OpenAI-compatible endpoint.

## Outputs

Training checkpoints and runtime outputs are written under the repository working directories such as `checkpoints/`, `outputs/`, and `.cache/`.

## Acknowledgement

This codebase builds on top of `ms-swift`, `verl`, and `transformers`.
