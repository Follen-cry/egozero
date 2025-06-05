#!/bin/bash


# ---- variables to modify here ----

export CUDA_VISIBLE_DEVICES=1
data_dirs=(
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v1_vrs/preprocess"
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v2_vrs/preprocess"
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v3_vrs/preprocess"
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v4_vrs/preprocess"
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v5_vrs/preprocess"
    "/nas/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v6_vrs/preprocess"

)
experiment="sweep_board"

# -----------------------------------


# format the data dirs into cli list format for hydra
data_dirs=$(printf ',"%s"' "${data_dirs[@]}")
data_dirs="[${data_dirs:1}]"  # Remove the leading comma

cd point_policy/

HYDRA_FULL_ERROR=1 python train.py \
    agent=p3po \
    suite=aria \
    dataloader=aria \
    eval=false \
    suite/task/franka_env=$experiment \
    data_dir=$data_dirs \
    experiment=$experiment \
    temporal_agg=true \
    num_queries=10 \
    suite.history=true \
    suite.history_len=10 \
    dataloader.bc_dataset.subsample=2 \
    suite.action_type=delta

cd ../
