export JAX_TRACEBACK_FILTERING="off"

# tpuv3-8
N_DATA_PARALLEL=1
N_MODEL_PARALLEL=8

# tpuv4-8
# N_DATA_PARALLEL=1
# N_MODEL_PARALLEL=4

# tpuv4-32
# N_DATA_PARALLEL=4
# N_MODEL_PARALLEL=4


NAME=llama3_to_qwen2_tokenizer
python3 scripts/cross_tokenizer_distill.py \
    --config=configs/cross_tokenizer_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
    hypernet.architecture=identity \
    multitask_aggregation_fn=approx_gradmag_preserve_mag \
    train_model_mode=lora \
    model_lora_rank=64 \
    model_lora_alpha=64 \
    n_data_parallel=$N_DATA_PARALLEL \
    n_model_parallel=$N_MODEL_PARALLEL \
    steps=5000 \
    eval_interval=1000 \
    save_interval=1000 \
    data.batch_size=64 \
    optimizer.grad_acc_steps=4 \
    data.num_workers=16 \
    data.batch_size=64 \
    student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    student.tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
    target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
    num_workers=16 \
    name=$NAME