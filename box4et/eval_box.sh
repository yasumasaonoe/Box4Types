#!/bin/bash

# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=bbn

CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id box_${DATA_NAME}_test \
--reload_model_name box_${DATA_NAME} \
--load \
--model_type bert-large-uncased-whole-word-masking \
--mode test \
--goal $DATA_NAME \
--emb_type box \
--threshold 0.5 \
--mc_box_type CenterSigmoidBoxTensor \
--type_box_type CenterSigmoidBoxTensor \
--gumbel_beta=0.00036026463511690845 \
--inv_softplus_temp=1.2471085395024732 \
--softplus_scale 1.0 \
--box_dim=109 \
--proj_layer highway \
--per_gpu_eval_batch_size 8 \
--eval_data ${DATA_NAME}/${DATA_NAME}_test.json