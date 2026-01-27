#!/bin/bash
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

export MUJOCO_GL=egl

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-infOPE/inf_ope/src/inf_ope/external/robomimic_infope
conda activate ope2


python robomimic/scripts/train_policy_robomimic.py \
    --config robomimic/exps/templates/diffusion_policy_image.json \
    --dataset /scr/shared/datasets/robomimic0.5/lift/mh/image_v15.hdf5