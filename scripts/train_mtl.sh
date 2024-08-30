#!/bin/bash


for data in "fluencybank" "sep28k"
do
    for model in "lstm" "convlstm" 
    do
        python ../train.py \
        --data_config "../baseline/configs/data/$data.yml" \
        --model_config "../baseline/configs/model/$model.yml" \
        --tasks "t1" "t2" \
        --losses "ce" "ccc" 
    done
done