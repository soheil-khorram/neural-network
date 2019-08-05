#!/bin/bash

# Author: Soheil Khorram
# License: Simplified BSD


export MODEL='model.dilated_conv'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
skip=0
dilated_layer_num=(4)
feature_layer_num=(5 7 9)
kernel_num=512
kernel_size=3
step_size=0.0001
epoch_num=20
batch_size=16

for dln in "${dilated_layer_num[@]}"; do
    for fln in "${feature_layer_num[@]}"; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/dln_${dln}_fln_${fln}_skip_${skip}/
        command="\
            python ../main.py \
                -skip $skip \
                -dilated-layer-num $dln \
                -feature-layer-num $fln\
                -kernel-num $kernel_num \
                -kernel-size $kernel_size \
                -step-size $step_size \
                -epoch-num $epoch_num \
                -batch-size $batch_size
                -net-summary-path $out_dir/net_summary.txt \
                -out-dir $out_dir"
        echo -e $command
        $command
    done
done
