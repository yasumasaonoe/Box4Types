#!/bin/bash

# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=onto
export MODEL_ID=vec_onto_test_training

if [[ $DATA_NAME == "ufet" ]]; then
    #export TRAIN_DATA=ufet/ufet_train.json
    export TRAIN_DATA=ufet/ufet_train.json,ufet/ufet_train_open_denoised.json,ufet/ufet_train_el_denoised.json
elif [[ $DATA_NAME == "onto" ]]; then
    export TRAIN_DATA=onto/onto_train.json
elif [[ $DATA_NAME == "figer" ]]; then
    export TRAIN_DATA=figer/figer_train.json
elif [[ $DATA_NAME == "bbn" ]]; then
    export TRAIN_DATA=bbn/bbn_train.json
else
    exit $exit_code
fi

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id=$MODEL_ID \
--goal=$DATA_NAME \
--seed=0 \
--train_data=$TRAIN_DATA \
--dev_data=${DATA_NAME}/${DATA_NAME}_dev.json \
--log_period=10 \
--eval_after=9 \
--eval_period=10 \
--emb_type=baseline \
--gradient_accumulation_steps=16 \
--learning_rate_cls=0.005392818468746842 \
--learning_rate_enc=2e-05 \
--mode=train \
--model_type=bert-large-uncased-whole-word-masking \
--num_epoch=100 \
--per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--proj_layer=highway \
--reduced_type_emb_dim=307 \
--save_period=10000000 \
--threshold=0.5