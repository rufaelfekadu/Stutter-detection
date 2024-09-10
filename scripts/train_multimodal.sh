ann="bau"

for c in 'P' 'B' "V" "FG" "HM" "any";do
    python train.py --model_config ./baseline/configs/model/multimodal_binary.yml --data_config ./baseline/configs/data/fluencybankvivit.yml \
        --opts data.name ${ann}_${c}  data.annotation ${c} data.label_path outputs/fluencybank/dataset/stutter_hf/label_split/${ann}_multimodal_ model.output_size 1 solver.batch_size 15 --logger
done