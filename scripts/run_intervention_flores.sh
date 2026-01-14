#!/bin/bash

# Default settings
SEED=2025 
K=15
THR=0.4
INPUT_DIR="routing"
MODEL_PATH="Qwen3-30B-A3B"

LANGS=("bn" "de" "en" "es" "fr" "ja" "sw" "zh")

# language for intervention
TARGET_LANGS=('ben_Beng' 'deu_Latn' 'spa_Latn' 'fra_Latn' 'jpn_Jpan' 'swh_Latn' 'eng_Latn' 'zho_Hans')

# Intervention layer ranges (format "start:end")
LAYER_RANGES=("0:-1" "0:4" "43:47" "22:26")

# Environment variables
export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES=1

# Loop 1: Iterate through each layer range
for RANGE in "${LAYER_RANGES[@]}"; do
    # Extract start and end layers using IFS or cut
    START_LAYER=$(echo $RANGE | cut -d':' -f1)
    END_LAYER=$(echo $RANGE | cut -d':' -f2)

    echo "=========================================================="
    echo "LAYER RANGE: Start=$START_LAYER, End=$END_LAYER"
    echo "=========================================================="

    # Loop 2: Iterate through every evaluation language
    for LANG in "${LANGS[@]}"; do
        
        # Loop 3: Iterate through every target language (Full Cross-Product)
        for TARGET_LANG in "${TARGET_LANGS[@]}"; do
            
            echo "[Running] Lang: $LANG | Target: $TARGET_LANG | Layers: $START_LAYER-$END_LAYER"

            python3 -m intervention.vllm_intervention_flores \
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
done