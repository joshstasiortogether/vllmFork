#!/bin/bash

# vLLM LongBench Test Script
# Usage: bash vllm_long_test.sh [GPU_ID] [MODEL_NAME] [TENSOR_PARALLEL_SIZE] [GPU_MEMORY_UTIL]

# Default values
GPU_ID=${1:-0}
MODEL_NAME=${2:-"meta-llama/Llama-2-7b-hf"}
TENSOR_PARALLEL_SIZE=${3:-1}
GPU_MEMORY_UTIL=${4:-0.9}

echo "Running vLLM LongBench evaluation..."
echo "GPU ID: $GPU_ID"
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "=========================="

export CUDA_VISIBLE_DEVICES=$GPU_ID

python vllm_longbench_test.py \
    --model_name_or_path "$MODEL_NAME" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTIL \
    --output_dir pred_vllm

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Evaluation completed successfully!"
    echo "Running evaluation metrics..."
    
    # Extract model name for evaluation
    MODEL_EVAL_NAME=$(basename "$MODEL_NAME")
    
    python eval_longbench_vllm.py \
        --model "$MODEL_EVAL_NAME" \
        --output_dir pred_vllm
    
    echo ""
    echo "Results saved in pred_vllm/$MODEL_EVAL_NAME/"
    echo "Check result.json for final scores"
else
    echo "Evaluation failed!"
    exit 1
fi 