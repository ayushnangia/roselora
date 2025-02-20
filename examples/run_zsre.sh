#!/bin/bash

# Default hparams
base_model_list=llama2
editors=RoseLoRA

sequentials=1
size=-1
retrain=0
cuda=0

while getopts g:e:n:s:r:m: flag
do 
    case "${flag}" in 
        g) cuda=${OPTARG};;
        e) editors=$(echo ${OPTARG} | tr "," "\n");;
        n) size=${OPTARG};;
        s) sequentials=$(echo ${OPTARG} | tr "," "\n");;
        r) retrain=${OPTARG};;
        m) base_model_list=$(echo ${OPTARG} | tr "," "\n");;
    esac
done 


export CUDA_VISIBLE_DEVICES=$cuda
export CUDA_LAUNCH_BLOCKING=1

# edit is our conda env. You can change it to your own env name.
echo $cuda
eval "$(conda shell.bash hook)"
conda activate edit

for base_models in $base_model_list; do

    if [[ $base_models = llama2 ]]; then
        base_model=llama-7b
    fi 

    echo RUN: $base_models
        
    for editor in $editors; do
        for sequential in $sequentials; do
            python run_zsre.py \
                --editing_method $editor \
                --ds_size $size \
                --sequential_edit $sequential \
                --retrain $retrain \
                --data_dir=./data/ZsRE \
                --hparams_dir=../hparams/$editor/$base_model \
                --base_model=$base_models \

            exit_code=$?
            if [[ $exit_code = 1 ]]; then
                exit 
            fi

            printf '\n\n\n'
        done
    done
done
