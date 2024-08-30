for ann in "A1" "A2" "A3";do
    echo "Training Vivit model with annotator $ann"
    python train.py --model_config ./baseline/configs/model/vivit.yml --opts data.annotator $ann 
done