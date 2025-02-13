#!/bin/sh
#SBATCH -J 125M
#SBATCH -o out_moe/%j.out
#SBATCH -A L00120230003
#SBATCH -w pgpu27
#SBATCH -p p-A800
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1



source /cm/shared/apps/anaconda3/etc/profile.d/conda.sh
conda activate modded_nanogpt

torchrun --standalone --nproc_per_node=2 train_moe.py

# torchrun --standalone --nproc_per_node=2 test.py
