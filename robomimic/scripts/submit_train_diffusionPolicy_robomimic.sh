#!/bin/bash
#SBATCH --job-name=train_dp_robomimic
#SBATCH --array=0-1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs_slurm/train_diffusionPolicy_%j.log
#SBATCH --error=/dev/null

export MUJOCO_GL=egl

source ~/miniconda3/etc/profile.d/conda.sh
cd /scr/yutaizho/code/p-infOPE/inf_ope/src/inf_ope/external/robomimic_infope
conda activate ope2

DATASETS=(
    # "/scr/shared/datasets/robomimic0.5/lift/ph/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/lift/mh/image_v15.hdf5"
    "/scr/shared/datasets/robomimic0.5/can/ph/image_v15.hdf5"
    "/scr/shared/datasets/robomimic0.5/can/mh/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/square/ph/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/square/mh/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/transport/ph/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/transport/mh/image_v15.hdf5"
    # "/scr/shared/datasets/robomimic0.5/tool_hang/ph/image_v15.hdf5"
)

dataset=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python robomimic/scripts/train_policy_robomimic.py \
    --config robomimic/exps/templates/diffusion_policy_image.json \
    --dataset $dataset