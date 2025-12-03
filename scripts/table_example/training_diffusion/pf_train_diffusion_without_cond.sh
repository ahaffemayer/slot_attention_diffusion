#!/bin/bash
#SBATCH --job-name="motion_diffusion"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --output=stdout_fit_motion_without_cond.txt
#SBATCH --error=stderr_fit_motion_without_cond.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

. ../../../../slot_diffusion/bin/activate

python ../../../src/conditional_diffusion_motion/diffusion_transformer/trainings/without_cond/fit_indirect_motion_with_obstacles_without_cond_box.py fit --config config/pf_config_box_without_cond.yaml