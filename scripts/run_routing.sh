#!/bin/bash

LANGS=("arb_Arab" "ben_Beng" "zho_Hans" "deu_Latn" "eng_Latn" "spa_Latn" "fra_Latn" "jpn_Jpan" "swh_Latn" "kor_Hang")
MODEL_PATH="Qwen3-30B-A3B"
OUTPUT_PATH="routing"
CUDA_VISIBLE_DEVICES=1

for LANG in "${LANGS[@]}"; do
    echo "===== Running language: $LANG ====="

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python3 -m routing.infer_batch_belebele \
        --model_path "$MODEL_PATH" \
        --lang "$LANG" \
        --output_path "$OUTPUT_PATH"
done