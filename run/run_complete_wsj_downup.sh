#!/bin/bash

# Author: Soheil Khorram
# License: Simplified BSD


export MODEL='model.down_up_net'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
hourglass_num=(5 3 1)
layer_num=(5 3)
kernel_num=512
kernel_size=(5 3)
epoch_num=20
step_size=0.0001
batch_size=8

for hgn in "${hourglass_num[@]}"; do
    for ks in "${kernel_size[@]}"; do
        for ln in ${layer_num[@]}; do
            out_dir=$base_path/$DATA_LOADER/$MODEL/hg_${hgn}_ks_${ks}_ln_${ln}/
            command="\
                python ../main.py \
                    -batch-size $batch_size \
                    -hourglass-num $hgn \
                    -out-dir $out_dir \
                    -net-summary-path $out_dir/net_summary.txt \
                    -layer-num $ln \
                    -kernel-size $ks \
                    -kernel-num  $kernel_num \
                    -epoch-num $epoch_num \
                    -step-size $step_size \
                    -down-up-num $ln"
            echo -e $command
            $command
        done
    done
done
