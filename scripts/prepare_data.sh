#!/bin/bash

# Define directories
FILE_PATH="datasets/fluencybank/our_annotations/reading"
OUTPUT_DIR="datasets/stutter-bank/reading"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

python tools/prepare_data.py --file_path "$FILE_PATH" --save_path "$OUTPUT_DIR"

LABEL_CSV="$OUTPUT_DIR/csv/labels.csv"
GRAN_CSV="$OUTPUT_DIR/csv/gran_total.csv"

python tools/create_gran.py --label_csv "$LABEL_CSV" --save_path "$GRAN_CSV"

python tools/compute_iaa.py --gran_csv "$GRAN_CSV" \
    --save_path "$OUTPUT_DIR/csv" \
    --item_col "itemID" \
    --annotator_col "annotatorID" \
    --label_col "labels" \
    --dist_fn "iou_dist" \


