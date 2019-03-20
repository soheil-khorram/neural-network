#!/bin/bash

export MODEL='model.recursive_conv_net'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
emb_layer_num=1
recursive_layer_num_out=4
recursive_layer_num_in=4
kernel_num=512
kernel_size=5
epoch_num=20
step_size=0.0001

out_dir=$base_path/$DATA_LOADER/$MODEL/
command="\
    python ../main.py \
        -out-dir $out_dir \
        -net-summary-path $out_dir/net_summary.txt \
        -emb-layer-num $emb_layer_num \
        -recursive-layer-num-out $recursive_layer_num_out \
        -recursive-layer-num-in $recursive_layer_num_in \
        -kernel-num $kernel_num \
        -kernel-size $kernel_size \
        -step-size $step_size \
        -epoch-num $epoch_num"
echo -e $command
$command

