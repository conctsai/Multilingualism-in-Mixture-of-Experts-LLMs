#!/bin/bash

python3 -m routing.infer_batch_mgsm

export VLLM_DISABLE_COMPILE_CACHE=1
export CUDA_VISIBLE_DEVICES=1


LANGS=("ja" "sw" "fr" "es" "bn" "de")

INPUT_DIR="routing/mgsm_result"
PATTERN="mgsm_{lang}.json"
K=20
THR=0.7
MODEL_PATH="../Qwen3-30B-A3B"
SEED=2025
STEER_START_LAYER=10
STEER_END_LAYER=39



TARGET_LANG="en,zh"
LAMBDA=0.022

for LANG in "${LANGS[@]}"; do
    echo "===== Steer coefficients source language: English, Optimized lambda: $LAMBDA, Target language: $LANG ====="

    python3 -m steer.vllm_steer_polymath \
        --input_dir "$INPUT_DIR" \
        --target_lang "$TARGET_LANG" \
        --k $K \
        --thr $THR \
        --lang "$LANG" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --steer_lambda $LAMBDA \
        --steer_start_layer $STEER_START_LAYER \
        --steer_end_layer $STEER_END_LAYER \
        --pattern "$PATTERN" \
        --shared

done


TARGET_LANG="zh,en"
LAMBDA=0.034

for LANG in "${LANGS[@]}"; do
    echo "===== Steer coefficients source language: Chinese, Optimized lambda: $LAMBDA, Target language: $LANG ====="

    python3 -m steer.vllm_steer_polymath \
        --input_dir "$INPUT_DIR" \
        --target_lang "$TARGET_LANG" \
        --k $K \
        --thr $THR \
        --lang "$LANG" \
        --model_path "$MODEL_PATH" \
        --seed $SEED \
        --steer_lambda $LAMBDA \
        --steer_start_layer $STEER_START_LAYER \
        --steer_end_layer $STEER_END_LAYER \
        --pattern "$PATTERN" \
        --shared

done


# TARGET_LANG="en,zh"
# LAMBDA=0.022
# STEER_START_LAYER=0
# STEER_END_LAYER=7

# for LANG in "${LANGS[@]}"; do
#     echo "===== Early layer ====="

#     python3 -m steer.vllm_steer_polymath \
#         --input_dir "$INPUT_DIR" \
#         --target_lang "$TARGET_LANG" \
#         --k $K \
#         --thr $THR \
#         --lang "$LANG" \
#         --model_path "$MODEL_PATH" \
#         --seed $SEED \
#         --steer_lambda $LAMBDA \
#         --steer_start_layer $STEER_START_LAYER \
#         --steer_end_layer $STEER_END_LAYER \
#         --pattern "$PATTERN" \
#         --shared

# done

# TARGET_LANG="en,zh"
# LAMBDA=0.022
# STEER_START_LAYER=40
# STEER_END_LAYER=47

# for LANG in "${LANGS[@]}"; do
#     echo "===== Late layer ====="

#     python3 -m steer.vllm_steer_polymath \
#         --input_dir "$INPUT_DIR" \
#         --target_lang "$TARGET_LANG" \
#         --k $K \
#         --thr $THR \
#         --lang "$LANG" \
#         --model_path "$MODEL_PATH" \
#         --seed $SEED \
#         --steer_lambda $LAMBDA \
#         --steer_start_layer $STEER_START_LAYER \
#         --steer_end_layer $STEER_END_LAYER \
#         --pattern "$PATTERN" \
#         --shared

# done
