#!/bin/bash

#SBATCH --job-name=stutter       # Job name
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --gres=gpu:1         # Number of GPUs (per node)
#SBATCH -p nlp-dept                   # Use the it-dept partition
#SBATCH -q nlp-pool
#SBATCH --exclude=gpu40-6
#SBATCH --time=48:00:00 


# for ann in "A1" "A2" "A3";do
#     echo "Training Vivit model with annotator $ann"
#     python train.py --model_config ./baseline/configs/model/vivit.yml --opts data.annotator $ann 
# done


# echo "Training Vivit model with annotator bau"
# python train.py --model_config ./baseline/configs/model/vivit.yml --data_config ./baseline/configs/data/fluencybankvivit.yml \
#     --opts data.name multilabel_sad  data.label_path outputs/fluencybank/dataset/stutter_hf/label_split/sad_multimodal_ model.output_size 3 --logger

# python train.py --model_config ./baseline/configs/model/vivit.yml --data_config ./baseline/configs/data/fluencybankvivit.yml \
#     --opts data.name sad_FG data.annotation FG  data.label_path outputs/fluencybank/dataset/stutter_hf/label_split/sad_multimodal_ model.output_size 1 --logger

# for ann in "bau" "sad";do
#     python train.py --model_config ./baseline/configs/model/vivit.yml --data_config ./baseline/configs/data/fluencybankvivit.yml \
#         --opts data.name ${ann}_any data.annotation any data.label_path outputs/fluencybank/dataset/stutter_hf/label_split/${ann}_multimodal_ solver.batch_size 20 --logger
# done

ann="sad"
labels=("any")

for c in ${labels[@]};do
    python train.py --model_config ./baseline/configs/model/vivit.yml --data_config ./baseline/configs/data/fluencybankvivit.yml \
        --opts data.name ${ann}_${c} data.annotation ${c} data.label_path outputs/fluencybank/dataset/stutter_hf/label_split/${ann}_multimodal_ solver.batch_size 20 solver.epochs 50 --logger
done