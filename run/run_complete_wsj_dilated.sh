#!/bin/bash

export MODEL='model.dilated_conv'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
layer_num=(5 6 4)
sub_layer_num=(2 1)
kernel_num=512
kernel_size=(3 5)
step_size=0.0001
epoch_num=20
batch_size=16

for ln in "${layer_num[@]}"; do
    for sln in "${sub_layer_num[@]}"; do
        for ks in ${kernel_size[@]}; do
            out_dir=$base_path/$DATA_LOADER/$MODEL/ln_${ln}_sln_${sln}_ks_${ks}/
            command="\
                python ../main.py \
                    -layer-num $ln \
                    -sub-layer-num $sln \
                    -kernel-num $kernel_num \
                    -kernel-size $ks \
                    -step-size $step_size \
                    -epoch-num $epoch_num \
                    -batch-size $batch_size
                    -net-summary-path $out_dir/net_summary.txt \
                    -out-dir $out_dir"
            echo -e $command
            $command
        done
    done
done