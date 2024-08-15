#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python scripts/reversal_suffix_elicitation.py --model_size 160m \
    --eval_size 100 --dataset allenai/real-toxicity-prompts \
    --vocab_batch_size 1000