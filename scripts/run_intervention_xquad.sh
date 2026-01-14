#!/bin/bash

MODEL_PATH="../Qwen3-30B-A3B"
INPUT_DIR="routing"
OUTPUT_PATH="routing"
LANGS=("ar" "el" "ru" "en" "tr")
TARGET_LANGS=("arb_Arab" "ell_Grek" "rus_Cyrl" "eng_Latn" "tur_Latn")
SEED=2025
THR=0.5
K=32

# get routing data of target languages
for LANG in "${TARGET_LANGS[@]}"; do
    echo "===== Running language: $LANG ====="

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    python3 -m routing.infer_batch_belebele \
        --model_path "$MODEL_PATH" \
        --lang "$LANG" \
        --output_path "$OUTPUT_PATH"
done

LAYER_RANGES=("0:-1" "0:9" "19:28" "38:47")

for RANGE in "${LAYER_RANGES[@]}"; do
    START_LAYER=${RANGE%:*}
    END_LAYER=${RANGE#*:}

    for (( i=0; i<${#LANGS[@]}; i++ )); do
        LANG=${LANGS[$i]}
        TARGET_LANG=${TARGET_LANGS[$i]}

        echo "----------------------------------------"
        echo "Evaluating pair: $LANG -> $TARGET_LANG"
        echo "Start layer: $START_LAYER, End layer: $END_LAYER"
        echo "K: $K, Threshold: $THR"
        echo "----------------------------------------"

        VLLM_DISABLE_COMPILE_CACHE=1 \
        CUDA_VISIBLE_DEVICES=1 \
        python3 -m intervention.vllm_intervention_xquad \
            --input_dir "$INPUT_DIR" \
            --target_lang "$TARGET_LANG" \
            --k $K \
            --thr $THR \
            --lang "$LANG" \
            --model_path "$MODEL_PATH" \
            --seed $SEED \
            --start_layer $START_LAYER \
            --end_layer $END_LAYER
    done      
done