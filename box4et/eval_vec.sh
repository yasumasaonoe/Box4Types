#!/bin/bash

# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=onto

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id vec_${DATA_NAME}_test \
--reload_model_name vec_${DATA_NAME} \
--load \
--model_type bert-large-uncased-whole-word-masking \
--mode test \
--goal $DATA_NAME \
--emb_type baseline \
--threshold 0.5 \
--reduced_type_emb_dim=307 \
--per_gpu_eval_batch_size 8 \
--eval_data ${DATA_NAME}/${DATA_NAME}_test.json