#!/usr/bin/env bash
set -euo pipefail

EFFOPD_VERL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export EFFOPD_VERL_ROOT
export PYTHONPATH="$EFFOPD_VERL_ROOT"
cd "$EFFOPD_VERL_ROOT/../.."
python3 "$EFFOPD_VERL_ROOT/scripts/check_effopd_import.py"
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export USED_MODEL="${USED_MODEL:-no_api}"
unset ROCR_VISIBLE_DEVICES

STUDENT_MODEL="${STUDENT_MODEL:-models/Qwen3-4B}"
TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-4B-Non-Thinking-RL-Math}"
TRAIN_FILE="${TRAIN_FILE:-data/DeepMath-103K/train_filtered_level6.parquet}"
VAL_FILES="${VAL_FILES:-['data/AIME2024/test.parquet','data/AIME2025/test.parquet']}"
QUICK_VAL_FILE="${QUICK_VAL_FILE:-data/quick_validation/math_validation.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/qwen3_4b_math}"

python3 -m verl.trainer.main_ppo \
        ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="$EFFOPD_VERL_ROOT" \
        algorithm.adv_estimator=grpo \
        algorithm.rollout_correction.rollout_is=token \
        algorithm.rollout_correction.rollout_is_threshold=5.0 \
        algorithm.rollout_correction.rollout_rs=null \
        algorithm.rollout_correction.bypass_mode=false \
        actor_rollout_ref.rollout.calculate_log_probs=true \
        data.train_files="$TRAIN_FILE" \
        data.val_files="$VAL_FILES" \
        data.train_batch_size=1024 \
        data.max_prompt_length=2048 \
        data.max_response_length=16384 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.shuffle=True \
        data.return_raw_chat=True \
        +data.apply_chat_template_kwargs.enable_thinking=False \
        actor_rollout_ref.model.path="$STUDENT_MODEL" \
        +actor_rollout_ref.ref.model.path="$TEACHER_MODEL" \
        +actor_rollout_ref.model.base_model_path="$STUDENT_MODEL" \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=22000 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.max_num_batched_tokens=22000 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.n=32 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        reward_model.reward_manager=naive \
        trainer.critic_warmup=0 \
        trainer.val_before_train=True \
        trainer.logger='["console","wandb"]' \
        trainer.log_val_generations=10 \
        trainer.project_name='on-policy-distillation' \
        trainer.experiment_name=qwen3_4b_math \
        trainer.n_gpus_per_node="${N_GPUS:-8}" \
        trainer.nnodes=1 \
        trainer.save_freq=2 \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.test_freq=2 \
        trainer.total_epochs=3 \
        trainer.total_training_steps=null \
        data.train_max_samples=-1 \
        data.val_max_samples=-1 \
        trainer.enable_iterative_test=True \
        trainer.max_test_iterations=5 \
        data.iterative_test_files="$QUICK_VAL_FILE" \
        data.iterative_test_batch_size=null \
        data.iterative_test_max_samples=-1 "$@"
