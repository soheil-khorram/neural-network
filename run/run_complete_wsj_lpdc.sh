#!/bin/bash

# Author: Soheil Khorram
# License: Simplified BSD


export MODEL='model.lpdc'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num_ls=(4)
lpdc_depth_ls=(4)
kernel_num=512
kernel_size=3
step_size=0.0001
epoch_num=20
batch_size=16

for lpdc_num in "${lpdc_num_ls[@]}"; do
    for lpdc_depth in "${lpdc_depth_ls[@]}"; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}/
        command="\
            python ../main.py \
                -lpdc-num $lpdc_num \
                -lpdc-depth $lpdc_depth \
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



base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num_ls=(5)
lpdc_depth_ls=(5)
kernel_num=512
kernel_size=3
step_size=0.0001
epoch_num=20
batch_size=8

for lpdc_num in "${lpdc_num_ls[@]}"; do
    for lpdc_depth in "${lpdc_depth_ls[@]}"; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}/
        command="\
            python ../main.py \
                -lpdc-num $lpdc_num \
                -lpdc-depth $lpdc_depth \
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



base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num_ls=(5)
lpdc_depth_ls=(3)
kernel_num=512
kernel_size=3
step_size=0.0001
epoch_num=20
batch_size=16

for lpdc_num in "${lpdc_num_ls[@]}"; do
    for lpdc_depth in "${lpdc_depth_ls[@]}"; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}/
        command="\
            python ../main.py \
                -lpdc-num $lpdc_num \
                -lpdc-depth $lpdc_depth \
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

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num_ls=(3)
lpdc_depth_ls=(5)
kernel_num=512
kernel_size=3
step_size=0.0001
epoch_num=20
batch_size=16

for lpdc_num in "${lpdc_num_ls[@]}"; do
    for lpdc_depth in "${lpdc_depth_ls[@]}"; do
        out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}/
        command="\
            python ../main.py \
                -lpdc-num $lpdc_num \
                -lpdc-depth $lpdc_depth \
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

