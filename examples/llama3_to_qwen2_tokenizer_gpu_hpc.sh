#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J gpujob
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MLMI-set56-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=10:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! ######################################################################################
#! ######################################################################################

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

source /home/${USER}/.bashrc
source /home/${USER}/miniforge3/bin/activate
mamba activate "/home/${USER}/miniforge3/envs/tokenkit"

export JAX_TRACEBACK_FILTERING="off"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/set56/miniforge3/envs/tokenkit/lib"

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
    n_data_parallel=2 \
    n_model_parallel=2 \
    steps=5000 \
    eval_interval=1000 \
    save_interval=1000 \
    data.batch_size=64 \
    optimizer.grad_acc_steps=4 \
    data.num_workers=0 \
    data.batch_size=64 \
    student.pretrained_model_name_or_path="benjamin/Llama-3.2-3B-Instruct-flax" \
    student.tokenizer_name=\'meta-llama/Llama-3.2-3B-Instruct:source=Llama3\' \
    target_tokenizer_name=\'Qwen/Qwen2.5-1.5B:source=Qwen2:target=Llama3\' \
    num_workers=0 \
    name=$NAME