#!/bin/bash


# ---- variables to modify here ----

export CUDA_VISIBLE_DEVICES=0
data_dirs=(
    "/nas/projectaria/mps/sweep_board_28/mps_sweep-board-v1-28_vrs/preprocess"
    "/nas/projectaria/mps/sweep_board_28/mps_sweep-board-v2-28_vrs/preprocess"
    "/nas/projectaria/mps/sweep_board_28/mps_sweep-board-v3-28_vrs/preprocess"
    "/nas/projectaria/mps/sweep_board_28/mps_sweep-board-v4-28_vrs/preprocess"
    "/nas/projectaria/mps/sweep_board_28/mps_sweep-board-v5-28_vrs/preprocess"
)
experiment="sweep_board"

bc_weight="/home/david/egozero/point_policy/exp_local/2025.05.09/put_book/deterministic/194215_hidden_dim_512/snapshot/320000.pt"

# -----------------------------------


if [[ ! -f "$bc_weight" ]]; then
    echo "Error: BC checkpoint '$bc_weight' not found!" >&2
    exit 1
fi


# format the data dirs into cli list format for hydra
data_dirs=$(printf ',"%s"' "${data_dirs[@]}")
data_dirs="[${data_dirs:1}]"  # Remove the leading comma

cd point_policy/

HYDRA_FULL_ERROR=1 OC_CAUSE=1 python eval_point_track.py \
    agent=p3po \
    suite=aria \
    dataloader=aria \
    eval=true \
    suite/task/franka_env=$experiment \
    data_dir=$data_dirs \
    experiment=eval_$experiment \
    bc_weight=$bc_weight \
    temporal_agg=true \
    num_queries=10 \
    suite.history=true \
    suite.history_len=10

cd ../
