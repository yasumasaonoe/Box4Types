#!/bin/bash

# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=ufet
export MODEL_ID=box_ufet_0

if [[ $DATA_NAME == "ufet" ]]; then
    export TRAIN_DATA=ufet/ufet_train.json
    #export TRAIN_DATA=ufet/ufet_train.json,ufet/ufet_train_open_denoised.json,ufet/ufet_train_el_denoised.json
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
--log_period=100 \
--eval_after=99 \
--eval_period=100 \
--alpha_type_reg=0 \
--box_dim=109 \
--box_offset=0.5 \
--emb_type=box \
--gradient_accumulation_steps=16 \
--gumbel_beta=0.00036026463511690845 \
--inv_softplus_temp=1.2471085395024732 \
--learning_rate_cls=0.003720789473794256 \
--learning_rate_enc=2e-05 \
--mc_box_type=CenterSigmoidBoxTensor \
--mode=train \
--model_type=bert-large-uncased-whole-word-masking \
--n_negatives=1000 \
--num_epoch=100 \
--per_gpu_eval_batch_size=8 \
--per_gpu_train_batch_size=8 \
--proj_layer=highway \
--save_period=10000000 \
--softplus_scale=1 \
--th_type_vol=0 \
--threshold=0.5 \
--type_box_type=CenterSigmoidBoxTensor \
--use_gumbel_baysian=True