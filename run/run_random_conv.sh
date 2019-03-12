#!/bin/bash

export MODEL='model.conv_net'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0

base_path=$(pwd)
base_path=$base_path"/../../exp_results_random"
mkdir -p $base_path
layer_num=(10 8 6)
kernel_num=512
kernel_size=(9 7 5)
epoch_num=10
step_size=0.0001

while true; do
    ks=${kernel_size[$((RANDOM % ${#kernel_size[@]}))]}
    ln=${layer_num[$((RANDOM % ${#layer_num[@]}))]}
    out_dir=$base_path/$DATA_LOADER/$MODEL/ks_${ks}_ln_${ln}/
    command="\
        python ../main.py \
            -out-dir $out_dir \
            -layer-num $ln \
            -kernel-size $ks \
            -kernel-num  $kernel_num \
            -epoch-num $epoch_num \
            -step-size $step_size"
    echo -e $command
    $command
done
