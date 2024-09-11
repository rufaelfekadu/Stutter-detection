#!/bin/bash

root="datasets/fluencybank/ds_label/reading"

for ann in "A1" "A2" "A3" "bau" "mas" "sad"
do
    for model in "lstm"
    do
        python train.py \
        --data_config "baseline/configs/data/fluencybank.yml" \
        --model_config "baseline/configs/model/$model.yml" \
        --logger \
        --opts \
        data.root "$root/$ann/clips/audio" \
        data.label_path "$root/$ann/total_label.csv" \
        data.cache_dir "$root" \
        data.annotator "$ann" \
        output.save_dir "outputs/$ann"
    done
done
