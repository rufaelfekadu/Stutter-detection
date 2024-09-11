
for annotator in "ds5"; do
    file_list="tools/list_${annotator}.txt"
    python evaluate.py "$file_list" -o "results_${annotator}.yaml"

done