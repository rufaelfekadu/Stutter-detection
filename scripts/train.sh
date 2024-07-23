#!/bin/bash
data="fluencybank"
model="lstm"

python train.py \
--data_config "baseline/configs/data/$data.yml" \
--model_config "baseline/configs/model/$model.yml"