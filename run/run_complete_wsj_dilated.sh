#!/bin/bash

export MODEL='model.dilated_conv'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
skip=1
super_layer_num=(2 1)
layer_num=(6 5 4)
sub_layer_num=1
kernel_num=512
kernel_size=(5 3)
step_size=0.0001
epoch_num=15
batch_size=16

for ln in "${layer_num[@]}"; do
    for sln in "${super_layer_num[@]}"; do
        for ks in ${kernel_size[@]}; do
            out_dir=$base_path/$DATA_LOADER/$MODEL/ln_${ln}_supln_${sln}_subln_${sub_layer_num}_ks_${ks}/
            command="\
                python ../main.py \
                    -skip $skip \
                    -layer-num $ln \
                    -super-layer-num $sln\
                    -sub-layer-num $sub_layer_num \
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

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
skip=1
super_layer_num=(1)
layer_num=(6 5 4)
sub_layer_num=2
kernel_num=512
kernel_size=(5 3)
step_size=0.0001
epoch_num=15
batch_size=16

for ln in "${layer_num[@]}"; do
    for sln in "${super_layer_num[@]}"; do
        for ks in ${kernel_size[@]}; do
            out_dir=$base_path/$DATA_LOADER/$MODEL/ln_${ln}_supln_${sln}_subln_${sub_layer_num}_ks_${ks}/
            command="\
                python ../main.py \
                    -skip $skip \
                    -layer-num $ln \
                    -super-layer-num $sln\
                    -sub-layer-num $sub_layer_num \
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
