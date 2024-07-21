#!/bin/bash

# data="$1"
# model="$2"

# Execute the Python file with the provided arguments
for data in "fluencybank" "sep28k"
do
    for model in "lstm" "convlstm" 
    do
        python baseline/train.py --data_config "baseline/configs/data/$data.yml" --model_config "baseline/configs/model/$model.yml"
    done
done