export JAX_TRACEBACK_FILTERING="off"

NAME=gemma2_distill_from_openmath2-llama
python3 scripts/cross_tokenizer_distill.py \
    --config=configs/math_cross_tokenizer_distill.yaml \
    --overrides \
    losses=[sft,alm_unconstrained] \
    alm_mode=merge_by_space_prob+append_space \
    tokenizer_pair_bias_threshold=0.1 \
    max_teacher_length=1024 \
    max_student_length=1024 \
    n_data_parallel=1 \
    n_model_parallel=1 \
    steps=5000 \
    eval_interval=5000 \
    save_interval=5000 \
    optimizer.learning_rate=5.e-6 \
    optimizer.weight_decay=0.0 \
    optimizer.max_grad_norm=null \
    optimizer.grad_acc_steps=4 \
    eval.tasks=[math_500_openmath2,gsm8k_openmath2] \
    eval.lengths=[2048] \
    eval.tokens_per_batch=16384 \
    eval.chat_template_mode=direct_encode_no_force_eos \
    data.batch_size=64 \
    log_interval=10 \
    sync_interval=100 \
    use_chat_template=true \
    chat_template_mode=direct_encode \
    hypernet.architecture=identity \
    train_embeddings=true \
    train_model_mode=full \
    eval_at_step_zero=false \
    save_at_step_zero=false \
    skip_lm_eval=false \
    num_workers=24 \
    name=$NAME
