#!/bin/bash


# ---- variables to modify here ----

export CUDA_VISIBLE_DEVICES=1
data_dirs=(
    "/data/projectaria/mps/vrs_file2/sweep_board2_28/mps_sweep-board2-v3_vrs"
)
experiment="sweep_board"

# -----------------------------------


for data_dir in "${data_dirs[@]}"; do
    echo "Processing: $data_dir"

    python preprocess.py \
      --mps_sample_path "$data_dir" \
      --is_right_hand \
      --task "$experiment"

done
