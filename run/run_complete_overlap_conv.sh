#!/bin/bash

# Author: Soheil Khorram
# License: Simplified BSD


export MODEL='model.conv_net'
export DATA_LOADER='data_loader.overlap_data_loader'
export CUDA_VISIBLE_DEVICE=0

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
layer_num=(6 8 10)
kernel_num=256
kernel_size=(5 7 9)
epoch_num=50
step_size=0.0001

for ks in "${kernel_size[@]}"; do
    for ln in ${layer_num[@]}; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/ks_${ks}_ln_${ln}/
        command="\
            python ../main.py \
                -out-dir $out_dir \
                -net-summary-path $out_dir/net_summary.txt \
                -layer-num $ln \
                -kernel-size $ks \
                -kernel-num  $kernel_num \
                -epoch-num $epoch_num \
                -step-size $step_size"
        echo -e $command
        $command
    done
done
