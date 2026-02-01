#!/bin/bash

llm_judge_model="default_eval_config"
trial_num=4
EVAL_RESULT_SAVE_PATH="eval_results"
evaluated_model=$1
thread_num=$2

# initialize output path
OUTPUT_PATH="eval_data/${evaluated_model}"
if [ ! -d "$OUTPUT_PATH" ]; then
    mkdir -p $OUTPUT_PATH
    echo "Created directory: $OUTPUT_PATH"
else
    echo "Directory already exists: $OUTPUT_PATH"
fi

# copy the ground-truth tables
mkdir -p $OUTPUT_PATH/overall_20250916_tables
cp -r data/overall_20250916_tables/* ${OUTPUT_PATH}/overall_20250916_tables


python scripts/run_eval_batching.py \
    --query_path data/overall_20250916.jsonl \
    --answer_root_path ${OUTPUT_PATH}/overall_20250916_tables \
    --response_root $OUTPUT_PATH \
    --model_config_name $evaluated_model \
    --result_save_root $EVAL_RESULT_SAVE_PATH \
    --trial_num $trial_num \
    --thread_num $thread_num \
    --use_cache

