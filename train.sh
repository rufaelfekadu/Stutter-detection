#!/bin/bash
data="fluencybank"
model="lstm"

python baseline/train.py \
--data_config "baseline/configs/data/$data.yml" \
--model_config "baseline/configs/model/$model.yml"