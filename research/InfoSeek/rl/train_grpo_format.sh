export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS

PROJECT_NAME='DR'
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
EXPERIMENT_NAME=Test
TRAIN_FILES=path/to/your/training/data/InfoSeek.parquet
VAL_FILES=path/to/your/validation/data/InfoSeek.parquet



PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=30000 \
    data.max_response_length=768 \
    data.max_start_length=1024 \
    data.max_obs_length=1024 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.420 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/${EXPERIMENT_NAME} \
    reward_model.structure_format_score=0.0 \
    max_turns=6 \
    retriever.url="http://127.0.0.1:8010/retrieve" \
    retriever.topk=5 \
    retriever.dense_retrieval_enable=true \
    retriever.elastic_search_enable=false \
    retriever.elastic_search_topk=5 \
    2>&1 | tee ./logs/${EXPERIMENT_NAME}.log


