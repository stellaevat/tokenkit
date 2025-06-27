USER=set56
CACHE_DIR=/home/$USER/.cache/huggingface
CACHE_DIR_NEW=/home/$USER/rds/hpc-work/.cache/huggingface

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/miniforge3/lib
export HF_HOME=$CACHE_DIR_NEW
export TRANSFORMERS_CACHE=$CACHE_DIR_NEW
cp $CACHE_DIR/token $CACHE_DIR_NEW/token

export JAX_TRACEBACK_FILTERING="off"

NAME=llama3_to_qwen2_tokenizer
python3 -m scripts.cross_tokenizer_distill \
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
    n_data_parallel=1 \
    n_model_parallel=1 \
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