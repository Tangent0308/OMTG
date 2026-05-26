#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RL_ROOT="$PROJECT_ROOT/rl"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VERL_PYTHONPATH="${VERL_PYTHONPATH:-}"

export PYTHONPATH="${RL_ROOT}:${RL_ROOT}/verl${VERL_PYTHONPATH:+:${VERL_PYTHONPATH}}:${PYTHONPATH:-}"
export DEBUG_PRINT="${DEBUG_PRINT:-false}"
export RM_NAME="${RM_NAME:-${OMTG_JUDGE_MODEL:-Qwen3-30B-A3B}}"
export TG_REWARD_STRATEGY="${TG_REWARD_STRATEGY:-tiouformatf1cacccaptionlength}"

MODEL_PATH="${MODEL_PATH:?Please set MODEL_PATH to your SFT checkpoint or actor init model}"
TRAIN_PARQUET="${TRAIN_PARQUET:?Please set TRAIN_PARQUET to the RL train parquet}"
EVAL_PARQUET="${EVAL_PARQUET:?Please set EVAL_PARQUET to the RL eval parquet}"

project_name="${PROJECT_NAME:-omtg}"
exp_name="${EXP_NAME:-grpo_4b_caption_agent}"
default_local_dir="${DEFAULT_LOCAL_DIR:-$PROJECT_ROOT/outputs/rl/$exp_name}"
rollout_dir="${ROLLOUT_DIR:-$PROJECT_ROOT/.cache/rollout/$exp_name}"
val_dir="${VAL_DIR:-$PROJECT_ROOT/.cache/val/$exp_name}"

mkdir -p "$default_local_dir" "$rollout_dir" "$val_dir"

unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY

# vLLM's memory pool is incompatible with PyTorch expandable segments.
# Strip that flag if it is inherited from the shell or old training scripts.
sanitize_cuda_alloc_conf() {
    local conf="${1:-}"
    local cleaned

    if [ -z "$conf" ]; then
        return 0
    fi

    cleaned="$(printf '%s' "$conf" | sed -E 's/(^|,)expandable_segments:True(,|$)/\1/g; s/^,+//; s/,+$//; s/,,+/,/g')"
    if [ "$cleaned" != "$conf" ]; then
        echo "Removing incompatible PYTORCH_CUDA_ALLOC_CONF entry: expandable_segments:True"
    fi

    if [ -n "$cleaned" ]; then
        export PYTORCH_CUDA_ALLOC_CONF="$cleaned"
    else
        unset PYTORCH_CUDA_ALLOC_CONF
    fi
}

sanitize_cuda_alloc_conf "${PYTORCH_CUDA_ALLOC_CONF:-}"
export MASTER_PORT="${MASTER_PORT:-23333}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${ARNOLD_ID:-0}"
export NNODES="${ARNOLD_WORKER_NUM:-1}"
export NPROC_PER_NODE="${ARNOLD_WORKER_GPU:-8}"

adv_estimator="${ADV_ESTIMATOR:-grpo}"
use_kl_in_reward="${USE_KL_IN_REWARD:-False}"
kl_coef="${KL_COEF:-0.0}"
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef="${KL_LOSS_COEF:-0.001}"
kl_loss_type="${KL_LOSS_TYPE:-mse}"
clip_ratio_low="${CLIP_RATIO_LOW:-0.2}"
clip_ratio_high="${CLIP_RATIO_HIGH:-0.285}"

max_prompt_length="${MAX_PROMPT_LENGTH:-16384}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}"
sp_size="${SP_SIZE:-1}"
actor_ppo_max_token_len="${ACTOR_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 5))}"
infer_ppo_max_token_len="${INFER_PPO_MAX_TOKEN_LEN:-$(((max_prompt_length + max_response_length) * 10))}"

n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-$((train_prompt_mini_bsz * 4))}"

temperature="${TEMPERATURE:-1.0}"
top_p="${TOP_P:-0.8}"
top_k="${TOP_K:-20}"
val_top_p="${VAL_TOP_P:-0.8}"
val_temperature="${VAL_TEMPERATURE:-0.7}"

use_dynamic_bsz="${USE_DYNAMIC_BSZ:-True}"
infer_micro_batch_size="${INFER_MICRO_BATCH_SIZE:-null}"
train_micro_batch_size="${TRAIN_MICRO_BATCH_SIZE:-null}"
offload="${OFFLOAD:-True}"
strategy="${STRATEGY:-fsdp2}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}"
max_num_seqs="${MAX_NUM_SEQS:-256}"

custom_reward_function_path="${CUSTOM_REWARD_FUNCTION_PATH:-$RL_ROOT/omtg_reward/reward_fn.py}"
custom_reward_function_name="${CUSTOM_REWARD_FUNCTION_NAME:-batch_tg_reward_fn}"

echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "MODEL_PATH: $MODEL_PATH"
echo "TRAIN_PARQUET: $TRAIN_PARQUET"
echo "EVAL_PARQUET: $EVAL_PARQUET"
echo "JUDGE_MODEL: ${OMTG_JUDGE_MODEL:-$RM_NAME}"

"$PYTHON_BIN" -u -m verl.trainer.main_ppo \
    data.train_files="['$TRAIN_PARQUET']" \
    data.val_files="['$EVAL_PARQUET']" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.image_patch_size=16 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.train_batch_size="${train_prompt_bsz}" \
    data.dataloader_num_workers=1 \
    data.val_batch_size=144 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.norm_adv_by_std_in_grpo=False \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${use_dynamic_bsz}" \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_ppo_max_token_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${infer_ppo_max_token_len}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload="${offload}" \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5.0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${train_prompt_mini_bsz}" \
    actor_rollout_ref.actor.ppo_micro_batch_size="${train_micro_batch_size}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${offload}" \
    actor_rollout_ref.actor.strategy="${strategy}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${offload}" \
    actor_rollout_ref.ref.strategy="${strategy}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${offload}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.max_num_seqs="${max_num_seqs}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size="${infer_micro_batch_size}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
    actor_rollout_ref.rollout.temperature="${temperature}" \
    actor_rollout_ref.rollout.top_p="${top_p}" \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${val_temperature}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${val_top_p}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size="${infer_micro_batch_size}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${sp_size}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=batch \
    reward_model.launch_reward_fn_async=False \
    data.use_shm=False \
    actor_rollout_ref.model.use_shm=False \
    custom_reward_function.path="${custom_reward_function_path}" \
    custom_reward_function.name="${custom_reward_function_name}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=40 \
    trainer.total_epochs=1 \
    actor_rollout_ref.nccl_timeout=3600 \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.rollout_data_dir="${rollout_dir}" \
    trainer.validation_data_dir="${val_dir}" \
    trainer.balance_batch=False \
    trainer.resume_mode=auto
