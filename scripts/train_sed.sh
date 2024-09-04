

# !/bin/bash
python train.py --model_config baseline/configs/model/whisperyoho.yml \
                --data_config baseline/configs/data/fluencybanksed.yml \
                --logger \


pred_dir="datasets/fluencybank/ds_30/reading/label"
file_list="file_list.txt"
rm -f "$file_list"
# Find all _ref.txt files and create the list with tab delimiter
find "$pred_dir" -type f -name '*_pred.txt' | while read -r pred_file; do
    ref_file="${pred_file/_pred.txt/_ref.txt}"
    echo -e "$ref_file\t$pred_file" >> "$file_list"
done

python evaluate.py "$file_list" -o results.yaml