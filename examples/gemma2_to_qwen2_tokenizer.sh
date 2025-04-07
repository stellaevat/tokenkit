#!/bin/bash

# Before running this script, you'll need to obtain the tokenizer bias data via
# ipython --pdb scripts/compute_tokenizer_info.py -- \
#     teacher_tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2\' \
#     target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2\' \
#     output=\'outputs/tokenizer_data/gemma2_to_qwen2_new\'

if [ "$1" == "debug" ]; then
    export JAX_DISABLE_JIT=1
    export JAX_PLATFORMS=""
    export WANDB_MODE=disabled
    DEBUG=true
else
    DEBUG=false
fi

echo "DEBUG: $DEBUG"

NAME=gemma2_to_qwen2_tokenizer
ipython --pdb scripts/cross_tokenizer_distill.py -- \
    losses=[alm_greedy] \
    debug=$DEBUG \
    alm_mode="merge_by_space_prob+append_space" \
    alm_diff_fn=binary_ce \
    max_teacher_length=512 \
    max_student_length=512 \
    n_data_parallel=1 \
    n_model_parallel=8 \
    steps=5000 \
    eval_interval=1000 \
    save_interval=1000 \
    optimizer.learning_rate=1e-5 \
    optimizer.weight_decay=0.0 \
    optimizer.max_grad_norm=null \
    optimizer.grad_acc_steps=4 \
    sync_interval=10 \
    eval.tasks=[arc_easy,arc_challenge,piqa,hellaswag,boolq,arithmetic,mmlu] \
    eval.lengths=[128,256,512,1024,2048] \
    eval.tokens_per_batch=8192 \
    eval.add_bos=true \
    data=tulu3 \
    data.batch_size=64 \
    data.num_workers=16 \
    ppl_eval_data.batch_size=64 \
    +student.pretrained_model_name_or_path="benjamin/gemma-2-2b-it-flax" \
    +student.tokenizer_name=\'google/gemma-2-2b-it:source=Gemma2\' \
    +target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Gemma2\' \
    tokenizer_pair_data_path=\'outputs/tokenizer_data/gemma2_to_qwen2\' \
    num_workers=16 \
    name=$NAME