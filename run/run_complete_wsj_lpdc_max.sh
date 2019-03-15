#!/bin/bash

export MODEL='model.lpdc_max'
export DATA_LOADER='data_loader.wsj_data_loader'
export CUDA_VISIBLE_DEVICE=0


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num=5
lpdc_depth=4
kernel_size=3
pool_size=3
dilation_size=3
kernel_num=512
epoch_num=20
batch_size=8
step_size=0.0001


out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}_kernel_size_${kernel_size}_pool_size_${pool_size}_dilation_size_${dilation_size}/
command="\
    python ../main.py \
        -lpdc-num $lpdc_num \
        -lpdc-depth $lpdc_depth \
        -kernel-size $kernel_size \
        -pool-size $pool_size \
        -dilation-size $dilation_size \
        -kernel-num $kernel_num \
        -epoch-num $epoch_num \
        -batch-size $batch_size
        -step-size $step_size \
        -net-summary-path $out_dir/net_summary.txt \
        -out-dir $out_dir"
echo -e $command
$command


base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num=5
lpdc_depth=5
kernel_size=2
pool_size=2
dilation_size=2
kernel_num=512
epoch_num=20
batch_size=8
step_size=0.0001


out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}_kernel_size_${kernel_size}_pool_size_${pool_size}_dilation_size_${dilation_size}/
command="\
    python ../main.py \
        -lpdc-num $lpdc_num \
        -lpdc-depth $lpdc_depth \
        -kernel-size $kernel_size \
        -pool-size $pool_size \
        -dilation-size $dilation_size \
        -kernel-num $kernel_num \
        -epoch-num $epoch_num \
        -batch-size $batch_size
        -step-size $step_size \
        -net-summary-path $out_dir/net_summary.txt \
        -out-dir $out_dir"
echo -e $command
$command



base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num=5
lpdc_depth=3
kernel_size=3
pool_size=3
dilation_size=3
kernel_num=512
epoch_num=20
batch_size=8
step_size=0.0001


out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}_kernel_size_${kernel_size}_pool_size_${pool_size}_dilation_size_${dilation_size}/
command="\
    python ../main.py \
        -lpdc-num $lpdc_num \
        -lpdc-depth $lpdc_depth \
        -kernel-size $kernel_size \
        -pool-size $pool_size \
        -dilation-size $dilation_size \
        -kernel-num $kernel_num \
        -epoch-num $epoch_num \
        -batch-size $batch_size
        -step-size $step_size \
        -net-summary-path $out_dir/net_summary.txt \
        -out-dir $out_dir"
echo -e $command
$command

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num=5
lpdc_depth=4
kernel_size=2
pool_size=2
dilation_size=2
kernel_num=512
epoch_num=20
batch_size=8
step_size=0.0001


out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}_kernel_size_${kernel_size}_pool_size_${pool_size}_dilation_size_${dilation_size}/
command="\
    python ../main.py \
        -lpdc-num $lpdc_num \
        -lpdc-depth $lpdc_depth \
        -kernel-size $kernel_size \
        -pool-size $pool_size \
        -dilation-size $dilation_size \
        -kernel-num $kernel_num \
        -epoch-num $epoch_num \
        -batch-size $batch_size
        -step-size $step_size \
        -net-summary-path $out_dir/net_summary.txt \
        -out-dir $out_dir"
echo -e $command
$command

base_path=$(pwd)
base_path=$base_path"/../../exp_results"
mkdir -p $base_path
lpdc_num=5
lpdc_depth=3
kernel_size=2
pool_size=2
dilation_size=2
kernel_num=512
epoch_num=20
batch_size=8
step_size=0.0001


out_dir=$base_path/$DATA_LOADER/$MODEL/lpdc_num_${lpdc_num}_lpdc_depth_${lpdc_depth}_kernel_size_${kernel_size}_pool_size_${pool_size}_dilation_size_${dilation_size}/
command="\
    python ../main.py \
        -lpdc-num $lpdc_num \
        -lpdc-depth $lpdc_depth \
        -kernel-size $kernel_size \
        -pool-size $pool_size \
        -dilation-size $dilation_size \
        -kernel-num $kernel_num \
        -epoch-num $epoch_num \
        -batch-size $batch_size
        -step-size $step_size \
        -net-summary-path $out_dir/net_summary.txt \
        -out-dir $out_dir"
echo -e $command
$command
