## Entity Typing Training and Evaluation

Descriptions for selected arguments: 

| Flag                             | Description                                                                       |
|----------------------------------|-----------------------------------------------------------------------------------|
| `--model_id`                      | Experiment name.                                                                 |
| `--model_type`                    | Pretrained MLM type. Currently, BERT and RoBERTa are supported.                  |
| `--mode`                          | Whether to train or test. This can be either `train` or `test`.                  |
| `--goal`                          | Type vocab.                                                                      |
| `--learning_rate_enc`             | Initial learning rate for the Transformer encoder.                               |
| `--learning_rate_cls`             | Initial learning rate for the type embeddings.                                   |
| `--per_gpu_train_batch_size`      | The batch size per GPU in the train mode.                                        |
| `--per_gpu_eval_batch_size`       | The batch size per GPU in the eval mode.                                         |
| `--gradient_accumulation_steps`   | Number of updates steps to accumulate before performing a backward/update pass.  |
| `--log_period`                    | How often to save.                                                               |
| `--eval_period`                   | How often to run dev.                                                            |
| `--eval_after`                    | When to start to run dev.                                                        |
| `--save_period`                   | How often to save.                                                               |
| `--train_data`                    | Train data file pattern.                                                         |
| `--dev_data`                      | Dev data file pattern.                                                           |
  
Descriptions for all command line arguments are provided in `main.py`. 


### Training

Sample command for training on UFET:

```bash
# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=ufet
export MODEL_ID=box_onto_test_training
export TRAIN_DATA=ufet/ufet_train.json

$ CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id=$MODEL_ID \
--goal=$DATA_NAME \
--seed=0 \
--train_data=$TRAIN_DATA \
--dev_data=${DATA_NAME}/${DATA_NAME}_dev.json \
--log_period=10 \
--eval_after=9 \
--eval_period=10 \
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
```

### Evaluation

Sample command for testing on UFET:

```bash
# Select data from {ufet, onto, figer, bbn}
export DATA_NAME=ufet

$ CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--model_id box_${DATA_NAME}_dev \
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
--eval_data ${DATA_NAME}/${DATA_NAME}_dev.json
```