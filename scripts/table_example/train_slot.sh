#!/bin/bash
#SBATCH --job-name="slot"
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --output=stdout_slot.txt
#SBATCH --error=stderr_slot.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

. ../../../slot_diffusion/bin/activate
python ../../src/conditional_diffusion_motion/slot_attention/train.py