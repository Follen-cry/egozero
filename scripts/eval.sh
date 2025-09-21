#!/bin/bash


# ---- variables to modify here ----

export CUDA_VISIBLE_DEVICES=0
data_dirs=(
    "/ibex/user/cheny1b/egozero/mps_fold_clothes_2_vrs/preprocess"
)
experiment="fold_clothes"

bc_weight="/home/aloha/egozero_ws2/src/egozero/point_policy/exp_local/fold_clothes/deterministic/124159_hidden_dim_512/snapshot/100000.pt"
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
    suite/task/aloha_env=$experiment \
    data_dir=$data_dirs \
    experiment=eval_$experiment \
    bc_weight=$bc_weight \
    temporal_agg=true \
    num_queries=10 \
    suite.history=true \
    suite.history_len=10 \
    suite.T_ego_to_robot_yaml=/home/aloha/egozero_ws2/install/interbotix_xsarm_perception/share/interbotix_xsarm_perception/config/static_transforms.yaml

cd ../
