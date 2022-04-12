#!/usr/bin/env python3
import argparse
import gc
import json
import numpy as np
import os
import pickle
import random
import time
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from typing import Dict, Generator, List, Optional, Tuple, Union

# Import custom modules
import constant

from models import TransformerVecModel
from models import TransformerBoxModel
from data_utils import DatasetLoader
from data_utils import to_torch



"""
Args
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", help="Identifier for model")
parser.add_argument("--device", type=int, default=0, help="CUDA device")
parser.add_argument("--n_gpu", help="Number of GPUs.", type=int, default=1)
parser.add_argument("--mode",
                    help="Whether to train or test",
                    default="train",
                    choices=["train", "test"])
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="For distributed training: local_rank")

# Data
parser.add_argument("--train_data", help="Train data file pattern.", default="")
parser.add_argument("--dev_data", help="Dev data filename.", default="")
parser.add_argument("--eval_data", help="Test data filename.", default="")
parser.add_argument("--goal",
                    help="category vocab size.",
                    default="60k",
                    choices=["60k", "ufet", "5k", "1k", "onto", "figer",
                             "bbn", "toy"])
parser.add_argument("--seed", help="Random Seed", type=int, default=0)
parser.add_argument("--context_window_size",
                    help="Left and right context size.",
                    default=100)

# learning
parser.add_argument("--num_epoch",
                    help="The number of epoch.",
                    default=100,
                    type=int)
parser.add_argument("--per_gpu_train_batch_size",
                    help="The batch size per GPU",
                    default=8,
                    type=int)
parser.add_argument("--per_gpu_eval_batch_size",
                    help="The batch size per GPU.",
                    default=8,
                    type=int)
parser.add_argument("--learning_rate_enc",
                    help="BERT: start learning rate.",
                    default=2e-5,
                    type=float)
parser.add_argument("--learning_rate_cls",
                    help="Classifier: start learning rate.",
                    default=1e-3,
                    type=float)
parser.add_argument("--adam_epsilon_enc",
                    default=1e-8,
                    type=float,
                    help="BERT: Epsilon for Adam optimizer.")
parser.add_argument("--adam_epsilon_cls",
                    default=1e-8,
                    type=float,
                    help="Classifier: Epsilon for Adam optimizer.")
parser.add_argument("--hidden_dropout_prob",
                    help="Dropout rate",
                    default=.0,
                    type=float)
parser.add_argument("--n_negatives",
                    help="num of negative types during training.",
                    default=0,
                    type=int)
parser.add_argument("--neg_temp",
                    help="Temperature for softmax over negatives.",
                    default=.0,
                    type=float)
parser.add_argument("--gradient_accumulation_steps",
                    help="Number of updates steps to accumulate before "
                         "performing a backward/update pass.",
                    default=1,
                    type=int)
parser.add_argument("--alpha_l2_reg_cls",
                    help="Coefficient for L2 regularization on classifier "
                         "parameters.",
                    default=-1.,
                    type=float)
# -- Learning rate scheduler.
parser.add_argument("--use_scheduler_enc",
                    help="BERT: use lr scheduler.",
                    action='store_true')
parser.add_argument("--use_scheduler_cls",
                    help="Classifier: use lr scheduler.",
                    action='store_true')
parser.add_argument("--scheduler_type", help="Scheduler type.",
                    default="cyclic")
parser.add_argument("--step_size_up",
                    help="Step size up.",
                    default=2000,
                    type=int)
parser.add_argument("--warmup_steps",
                    help="Linear warmup over warmup_steps.",
                    default=0,
                    type=int)
parser.add_argument("--max_steps",
                    default=1000000,
                    type=int,
                    help="max # of steps.")

# Model
parser.add_argument("--model_type",
                    default="bert-large-uncased-whole-word-masking",
                    choices=[
                      "bert-base-uncased",
                      "bert-large-uncased",
                      "bert-large-uncased-whole-word-masking",
                      "roberta-base",
                      "roberta-large"
                    ])
parser.add_argument("--emb_type",
                    help="Embedding type.",
                    default="baseline",
                    choices=["baseline", "box"])
parser.add_argument("--mc_box_type",
                    help="Mention&Context box type.",
                    default="SigmoidBoxTensor")
parser.add_argument("--type_box_type",
                    help="Type box type.",
                    default="SigmoidBoxTensor")
parser.add_argument("--box_dim", help="Box dimension.", default=200, type=int)
parser.add_argument("--box_offset",
                    help="Offset for constant box.",
                    default=0.5,
                    type=float)
parser.add_argument("--threshold",
                    help="Threshold for type classification.",
                    default=0.5,
                    type=float)
parser.add_argument("--alpha_type_reg",
                    help="alpha_type_reg",
                    default=0.0,
                    type=float)
parser.add_argument("--th_type_vol",
                    help="th_type_reg",
                    default=0.0,
                    type=float)
parser.add_argument("--alpha_type_vol_l1",
                    help="alpha_type_vol_l1",
                    default=0.0,
                    type=float)
parser.add_argument("--alpha_type_vol_l2",
                    help="alpha_type_vol_l2",
                    default=0.0,
                    type=float)
parser.add_argument("--marginal_scale",
                    help="marghinal_scale",
                    default=1.0,
                    type=float)
parser.add_argument("--avg_pooling",
                    help="Averaging all hidden states instead of using [CLS].",
                    action='store_true')
parser.add_argument("--inv_softplus_temp",
                    help="Softplus temp.",
                    default=1.,
                    type=float)
parser.add_argument("--softplus_scale",
                    help="Softplus scale.",
                    default=1.,
                    type=float)
parser.add_argument("--proj_layer", help="Projection type.", default="linear")
parser.add_argument("--n_proj_layer", help="n_proj_layer", default=2, type=int)
parser.add_argument("--use_gumbel_baysian",
                    help="Make this Ture to use Gumbel box.",
                    default=False,
                    type=bool)
parser.add_argument("--gumbel_beta",
                    help="Gumbel beta.",
                    default=-1.,
                    type=float)
parser.add_argument("--reduced_type_emb_dim",
                    help="For baseline. Project down CLS vec.",
                    default=-1,
                    type=int)
parser.add_argument("--alpha_hierarchy_loss",
                    help="alpha_hierarchy_loss",
                    default=0.,
                    type=float)
parser.add_argument("--conditional_prob_path",
                    help="Type conditionals path.",
                    default="")
parser.add_argument("--marginal_prob_path",
                    help="Type marginals path.",
                    default="")
parser.add_argument("--encoder_layer_ids",
                    help="Encoder's hidden layer IDs that will be used for "
                         "the mentin and context representation. To use the "
                         "last 4 layers of BERT large, it should be "
                         "21 22 23 24, 0-indexed.",
                    nargs='*',
                    type=int)

# Save / log related
parser.add_argument("--save_period",
                    help="How often to save.",
                    default=1000,
                    type=int)
parser.add_argument("--save_best_model",
                    help="Save the current best model.",
                    default=True,
                    type=bool)
parser.add_argument("--eval_period",
                    help="How often to run dev.", default=500, type=int)
parser.add_argument("--log_period",
                    help="How often to record.",
                    default=1000,
                    type=int)
parser.add_argument("--eval_after",
                    help="Start eval after X steps.",
                    default=10,
                    type=int)
parser.add_argument("--max_num_eval",
                    help="Stop after evaluating X times.",
                    default=-1,
                    type=int)
parser.add_argument("--load", help="Load existing model.", action='store_true')
parser.add_argument("--reload_model_name",
                    help="Model name to reload.",
                    default="")
parser.add_argument("--pretrained_box_path",
                    help="Pretrained box path.",
                    default="")
parser.add_argument("--use_wandb",
                    help="Use Weights & Biases.",
                    default=False,
                    type=bool)
parser.add_argument("--wandb_project_name",
                    help="Weights & Biases: project name.",
                    default="")
parser.add_argument("--wandb_username",
                    help="Weights & Biases: username.",
                    default="")


"""
Utils
"""

SIGMOID = nn.Sigmoid()


def set_seed(seed: int, n_gpu: int):
  """Set seed for all random number generators."""
  if seed < 0:
    _seed = time.thread_time_ns() % 4294967295
  else:
    _seed = seed
  print("seed: {}".format(_seed))
  random.seed(_seed)
  np.random.seed(_seed)
  torch.manual_seed(_seed)
  if n_gpu > 0:
    torch.cuda.manual_seed_all(_seed)


def get_data_gen(dataname: str,
                 mode: str,
                 args: argparse.Namespace,
                 tokenizer: object) -> Generator[Dict, None, None]:
  """Returns a data generator."""
  data_path = constant.FILE_ROOT + dataname
  dataset = DatasetLoader(data_path, args, tokenizer)
  if mode == "train":
    data_gen = dataset.get_batch(args.train_batch_size,
                                 args.max_position_embeddings,
                                 args.num_epoch,
                                 eval_data=False)
  else:  # test mode
    data_gen = dataset.get_batch(args.eval_batch_size,
                                 args.max_position_embeddings,
                                 1,  # num epochs is 1.
                                 eval_data=True)
  return data_gen


def get_all_datasets(
  args: argparse.Namespace, tokenizer: object
) -> List[Generator[Dict, None, None]]:
  """Returens a list of training data geenrators."""
  train_gen_list = []
  if args.mode in ["train"]:
    for train_data_path in args.train_data.split(","):
      train_gen_list.append(
        get_data_gen(train_data_path, "train", args, tokenizer))
  return train_gen_list


def get_datasets(data_lists: List[Tuple[str, str]],
                 args: argparse.Namespace,
                 tokenizer: object) -> List[Generator[Dict, None, None]]:
  """Returens a list of dev/test data geenrators."""
  data_gen_list = []
  for dataname, mode in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, tokenizer))
  return data_gen_list


def evaluate_data(
  batch_num: int,
  dev_fname: str,
  model: Union[TransformerVecModel, TransformerBoxModel],
  id2word_dict: Dict[int, str],
  args: argparse.Namespace,
  device: torch.device
) -> Tuple[float, float, float, float, float, float, float, float]:
  """Computes macro&micro precision, recall, and F1."""
  model.eval()
  dev_gen = get_data_gen(dev_fname, "test", args, model.transformer_tokenizer)
  gold_pred = []
  eval_loss = 0.
  total_ex_count = 0
  for batch in tqdm(dev_gen):
    total_ex_count += len(batch["targets"])
    inputs, targets = to_torch(batch, device)
    loss, output_logits = model(inputs, targets)
    output_index = get_output_index(
      output_logits,
      threshold=args.threshold,
      is_prob=True if args.emb_type == "box" else False)
    gold_pred += get_gold_pred_str(output_index,
                                   batch["targets"].data.cpu().clone(),
                                   id2word_dict)
    eval_loss += loss.clone().item()
  eval_loss /= float(total_ex_count)
  count, pred_count, avg_pred_count, micro_p, micro_r, micro_f1 = micro(gold_pred)
  _, _, _, macro_p, macro_r, macro_f1 = macro(gold_pred)
  accuracy = sum(
    [set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  eval_loss_str = "Eval loss: {0:.7f} at step {1:d}".format(eval_loss,
                                                            batch_num)
  eval_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{" \
             "6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(count,
                                                        pred_count,
                                                        avg_pred_count,
                                                        micro_p,
                                                        micro_r,
                                                        micro_f1,
                                                        macro_p,
                                                        macro_r,
                                                        macro_f1)
  eval_str += "\t Dev EM: {0:.1f}%".format(accuracy * 100)
  print("==>  EVAL: seen " + repr(total_ex_count) + " examples.")
  print(eval_loss_str)
  print(gold_pred[:3])
  print("==> " + eval_str)
  model.train()
  dev_gen = None  # Delete a data generator.
  return eval_loss, macro_f1, macro_p, macro_r, micro_f1, micro_p, micro_r, \
         accuracy


def get_lr_scheduler(
  scheduler_type: str,
  optimizer: torch.optim.Optimizer,
  warmup_steps: Optional[int] = 0,
  max_steps: Optional[bool] = None,
  base_lr: float = 1e-4,
  max_lr: float = 1e-3,
  step_size_up: int = 2000
) -> torch.optim.lr_scheduler:
  """Returns lr scheduler."""
  if scheduler_type == "linear":
    return get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=max_steps)
  elif scheduler_type == "cos_hard_restart":
    return get_cosine_with_hard_restarts_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=max_steps,
      num_cycles=3)
  elif scheduler_type == "cyclic":
    return torch.optim.lr_scheduler.CyclicLR(
      optimizer,
      base_lr,
      max_lr,
      step_size_up=step_size_up,
      cycle_momentum=False)


def f1(p: float, r: float) -> float:
  if r == 0.:
    return 0.
  return 2 * p * r / float(p + r)


def macro(
  true_and_prediction: List[Tuple[List[str], List[str]]]
) -> Tuple[int, int, int, float, float, float]:
  """Computes macro precision, recall, and F1."""
  num_examples = len(true_and_prediction)
  p = 0.
  r = 0.
  pred_example_count = 0
  pred_label_count = 0.
  gold_label_count = 0.
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
      pred_label_count += len(predicted_labels)
      per_p = len(set(predicted_labels).intersection(set(true_labels))) / \
              float(len(predicted_labels))
      p += per_p
    if len(true_labels):
      gold_label_count += 1
      per_r = len(set(predicted_labels).intersection(set(true_labels))) / \
              float(len(true_labels))
      r += per_r
  if pred_example_count == 0 or gold_label_count == 0:
    return num_examples, 0, 0, 0., 0., 0.
  precision = p / float(pred_example_count)
  recall = r / gold_label_count
  avg_elem_per_pred = pred_label_count / float(pred_example_count)
  return num_examples, pred_example_count, avg_elem_per_pred, precision, \
         recall, f1(precision, recall)


def micro(
  true_and_prediction: List[Tuple[List[str], List[str]]]
) -> Tuple[int, int, int, float, float, float]:
  """Computes micro precision, recall, and F1."""
  num_examples = len(true_and_prediction)
  num_predicted_labels = 0.
  num_true_labels = 0.
  num_correct_labels = 0.
  pred_example_count = 0
  for true_labels, predicted_labels in true_and_prediction:
    if predicted_labels:
      pred_example_count += 1
    num_predicted_labels += len(predicted_labels)
    num_true_labels += len(true_labels)
    num_correct_labels += len(
      set(predicted_labels).intersection(set(true_labels)))
  if pred_example_count == 0 or num_predicted_labels == 0 \
          or num_true_labels == 0:
    return num_examples, 0, 0, 0., 0., 0.
  precision = num_correct_labels / num_predicted_labels
  recall = num_correct_labels / num_true_labels
  avg_elem_per_pred = num_predicted_labels / float(pred_example_count)
  return num_examples, pred_example_count, avg_elem_per_pred, precision, \
         recall, f1(precision, recall)


def load_model(reload_model_name: str,
               save_dir: str,
               model_id: str,
               model: Union[TransformerVecModel, TransformerBoxModel],
               optimizer_enc: Optional[torch.optim.Optimizer] = None,
               optimizer_cls: Optional[torch.optim.Optimizer] = None,
               scheduler_enc: Optional[object] = None,
               scheduler_cls: Optional[object] = None):
  """Loads a trained model."""
  if reload_model_name:
    model_file_name = "{0:s}/{1:s}.pt".format(save_dir, reload_model_name)
  else:
    model_file_name = "{0:s}/{1:s}.pt".format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint["state_dict"])
  if optimizer_enc and optimizer_cls:  # Continue training
    optimizer_enc.load_state_dict(checkpoint["optimizer_enc"])
    optimizer_cls.load_state_dict(checkpoint["optimizer_cls"])
  if scheduler_enc and scheduler_cls:
    scheduler_enc.load_state_dict(checkpoint["scheduler_enc"])
    scheduler_cls.load_state_dict(checkpoint["scheduler_cls"])
  else:  # Test
    total_params = 0
    # Log params
    for k in checkpoint["state_dict"]:
      elem = checkpoint["state_dict"][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      #print(k, elem.size())
      total_params += param_s
    param_str = ("Number of total parameters..{0:d}".format(total_params))
    print(param_str)
  print("Loading model from ... {0:s}".format(model_file_name))


def get_output_index(outputs: torch.Tensor,
                     threshold: float = 0.5,
                     is_prob: bool = False) -> List[List[int]]:
  """Given outputs from the decoder, generates prediction index."""
  pred_idx = []
  if is_prob:
    outputs = outputs.data.cpu().clone()
  else:
    outputs = SIGMOID(outputs).data.cpu().clone()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist))
       if single_dist[i] > threshold and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx


def get_gold_pred_str(
        pred_idx: List[List[int]],
        gold: List[List[int]],
        id2word_dict: Dict[int, str]
) -> List[Tuple[List[str], List[str]]]:
  """
  Given predicted ids and gold ids, generate a list of (gold, pred) pairs of
  length batch_size.
  """
  gold_strs = []
  for gold_i in gold:
    gold_strs.append([id2word_dict[i] for i in range(len(gold_i))
                      if gold_i[i] == 1])
  pred_strs = []
  for pred_idx1 in pred_idx:
    pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
  else:
    return list(zip(gold_strs, pred_strs))


def get_eval_string(true_prediction: List[Tuple[List[str], List[str]]]) -> str:
  """Returns an eval results string."""
  count, pred_count, avg_pred_count, p, r, f1 = micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{" \
               "6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(count,
                                                          pred_count,
                                                          avg_pred_count,
                                                          p,
                                                          r,
                                                          f1,
                                                          ma_p,
                                                          ma_r,
                                                          ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 \
             / len(true_prediction)
  output_str += "\t Dev EM: {0:.1f}%".format(accuracy * 100)
  return output_str


"""
Training 
"""

def _train(args: argparse.Namespace,
           model: Union[TransformerVecModel, TransformerBoxModel],
           device: torch.device):
  if args.use_wandb:
    # Use Weights and Biases
    print("==> Start training with Weights and Biases...")
    wandb.init(project=args.wandb_project_name,
               entity=args.wandb_username,
               name=args.model_id)
    wandb.config.update(args)
    wandb.watch(model)
    #save_model_to = wandb.run.dir
    save_model_to = os.path.join(constant.EXP_ROOT, args.model_id)
  else:
    print("==> Start training...")
    save_model_to = os.path.join(constant.EXP_ROOT, args.model_id)
  if not os.path.exists(save_model_to):
    print("==> Create {}".format(save_model_to))
    os.makedirs(save_model_to, exist_ok=False)
  print("==> Trained models will be saved at {}".format(save_model_to))
  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  print("==> Loading data generator... ")
  train_gen_list = get_all_datasets(args, model.transformer_tokenizer)
  print("==> Loading ID -> TYPE mapping... ")
  _word2id = constant.load_vocab_dict(constant.TYPE_FILES[args.goal])
  id2word_dict = {v: k for k, v in _word2id.items()}
  print("done. {} data gen(s)".format(len(train_gen_list)))
  print("Model Type: {}".format(args.model_type))
  total_loss = 0.
  batch_num = 0
  num_evals = 0
  best_macro_f1 = 0.
  start_time = time.time()
  init_time = time.time()
  print(
    "Total {} named params.".format(
      len([n for n, p in model.named_parameters()])))
  no_decay = ["bias", "LayerNorm.weight"]
  classifier_param_prefix = ["classifier",
                             "proj_layer",
                             "linear_projection",
                             "encoder_layer_proj"]
  encoder_parameters = [
    {
      "params": [p for n, p in model.named_parameters()
                 if not any(nd in n for nd in no_decay)
                 and not any(n.startswith(pre)
                             for pre in classifier_param_prefix)],
      "weight_decay": 0.0  #args.weight_decay,
    },
    {
      "params": [p for n, p in model.named_parameters()
                 if any(nd in n for nd in no_decay)
                 and not any(n.startswith(pre)
                             for pre in classifier_param_prefix)],
      "weight_decay": 0.0
    },
  ]
  classifier_parameters = [
    {
      "params": [p for n, p in model.named_parameters()
                 if any(n.startswith(pre) for pre in classifier_param_prefix)],
      "weight_decay": 0.0
    },
  ]
  classifier_parameter_names = set(
    [n for n, p in model.named_parameters()
     if any(n.startswith(pre) for pre in classifier_param_prefix)])
  print(
    "Encoder {}, Classifier {}".format(
      sum([len(p["params"]) for p in encoder_parameters]),
      sum([len(p["params"]) for p in classifier_parameters])
    )
  )
  print("classifier_parameters:", [n for n, p in model.named_parameters()
                                   if any(n.startswith(pre)
                                          for pre in classifier_param_prefix)])
  optimizer_enc = AdamW(encoder_parameters,
                        lr=args.learning_rate_enc,
                        eps=args.adam_epsilon_enc)
  optimizer_cls = AdamW(classifier_parameters,
                        lr=args.learning_rate_cls,
                        eps=args.adam_epsilon_cls)
  scheduler_enc = None
  if args.use_scheduler_enc:
    scheduler_enc = get_lr_scheduler(
      args.scheduler_type,
      optimizer_enc,
      warmup_steps=args.warmup_steps,
      max_steps=args.max_steps,
      base_lr=args.learning_rate_enc / 2.,
      max_lr=args.learning_rate_enc,
      step_size_up=int(args.step_size_up / args.gradient_accumulation_steps))
    print("Using lr scheduler (enc): base_lr={}, max_lr={}, "
          "step_size_up={}".format(
      args.learning_rate_enc / 2.,
      args.learning_rate_enc,
      int(args.step_size_up / args.gradient_accumulation_steps)))

  scheduler_cls = None
  if args.use_scheduler_cls:
    scheduler_cls = get_lr_scheduler(
      args.scheduler_type,
      optimizer_enc,
      warmup_steps=args.warmup_steps,
      max_steps=args.max_steps,
      base_lr=args.learning_rate_cls / 10.,
      max_lr=args.learning_rate_cls,
      step_size_up=int(args.step_size_up / args.gradient_accumulation_steps))
    print("Using lr scheduler (cls): base_lr={}, max_lr={}, "
          "step_size_up={}".format(
      args.learning_rate_cls / 10.,
      args.learning_rate_cls,
      int(args.step_size_up / args.gradient_accumulation_steps)))
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
  if args.load:
    load_model(args.reload_model_name,
               constant.EXP_ROOT,
               args.model_id,
               model,
               optimizer_enc,
               optimizer_cls)
  optimizer_enc.zero_grad()
  optimizer_cls.zero_grad()
  set_seed(args.seed, args.n_gpu)
  continue_training = True
  while continue_training:
    batch_num += 1  # single batch composed of all train signal passed by.
    for data_gen in train_gen_list:
      try:
        batch = next(data_gen)
        inputs, targets = to_torch(batch, device)
      except StopIteration:
        print("Done!")
        torch.save(
          {
            "state_dict": model.state_dict(),
            "optimizer_cls": optimizer_cls.state_dict(),
            "optimizer_enc": optimizer_enc.state_dict(),
            "scheduler_cls": scheduler_cls.state_dict()
            if scheduler_cls is not None else None,
            "scheduler_enc": scheduler_enc.state_dict()
            if scheduler_enc is not None else None,
            "args": args
          },
          "{0:s}/{1:s}.pt".format(save_model_to, args.model_id)
        )
        return
      model.train()

      #with autograd.detect_anomaly():
      if True:
        if args.model_type != "distilbert":
          inputs["token_type_ids"] = (
            batch["token_type_ids"] if args.model_type in ["bert",
                                                           "xlnet",
                                                           "albert"] else None
          )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_id
        loss, output_logits = model(inputs,
                                    targets,
                                    batch_num=batch_num)

        if args.alpha_l2_reg_cls > 0.:
          l2_reg = torch.tensor(0., device=args.device)
          for n, p in model.named_parameters():
            if n in classifier_parameter_names:
              l2_reg += torch.norm(p)
          loss += args.alpha_l2_reg_cls * l2_reg

        if args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item()

      if batch_num % args.gradient_accumulation_steps == 0:
        if torch.isnan(loss).any():
          print("WARNING: Loss is nan at update: {}.".format(batch_num))
          print(loss)
          continue
        else:
          optimizer_enc.step()
          optimizer_cls.step()
          if args.use_scheduler_enc:
            scheduler_enc.step()
          if args.use_scheduler_cls:
            scheduler_cls.step()

        optimizer_enc.zero_grad()
        optimizer_cls.zero_grad()

        if batch_num % args.log_period == 0 and batch_num > 0:
          gc.collect()
          cur_loss = float(1.0 * loss.clone().item())
          elapsed = time.time() - start_time
          train_loss_str = (
            "|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch".format(
              cur_loss, batch_num, elapsed * 1000 / args.log_period))
          start_time = time.time()
          print(train_loss_str)

        if batch_num % args.eval_period == 0 and batch_num > 0:
          output_index = get_output_index(
            output_logits,
            threshold=args.threshold,
            is_prob=True if args.emb_type == "box" else False)
          gold_pred_train = get_gold_pred_str(
            output_index,
              batch["targets"].data.cpu().clone(),
              id2word_dict)
          print(gold_pred_train[:10])
          accuracy = sum(
            [set(y) == set(yp) for y, yp in gold_pred_train]
          ) * 1.0 / len(gold_pred_train)
          train_acc_str = "==> Train EM: {0:.1f}%".format(accuracy * 100)
          print(train_acc_str)

    if batch_num % args.eval_period == 0 and batch_num > args.eval_after:
      # Evaluate Loss on the Turk Dev dataset.
      print("---- eval at step {0:d} ---".format(batch_num))
      eval_loss, macro_f1, macro_p, macro_r,\
      micro_f1, micro_p, micro_r, accuracy = \
        evaluate_data(
            batch_num, args.dev_data, model, id2word_dict, args, device)

      if args.use_wandb:
        cur_loss = float(1.0 * loss.clone().item())
        wandb.log(
          {
            "Dev Mi-P": micro_p,
            "Dev Mi-R": micro_r,
            "Dev Mi-F1": micro_f1,
            "Dev Ma-P": macro_p,
            "Dev Ma-R": macro_r,
            "Dev Ma-F1": macro_f1,
            "Dev Acc": accuracy * 100,
            "Dev Loss": eval_loss,
            "Train Loss": cur_loss
        })

      if args.save_best_model and best_macro_f1 < macro_f1:
        best_macro_f1 = macro_f1
        save_fname = "{0:s}/{1:s}_best.pt".format(save_model_to, args.model_id)
        torch.save(
          {
            "state_dict": model.state_dict(),
            "optimizer_cls": optimizer_cls.state_dict(),
            "optimizer_enc": optimizer_enc.state_dict(),
            "scheduler_cls": scheduler_cls.state_dict()
            if scheduler_cls is not None else None,
            "scheduler_enc": scheduler_enc.state_dict()
            if scheduler_enc is not None else None,
            "args": args
          },
          save_fname
        )
        print(
          "Total {0:.2f} minutes have passed, saving at {1:s} ".format(
            (time.time() - init_time) / 60, save_fname))

      num_evals += 1
      if 0 < args.max_num_eval == num_evals:
        print("Evaluated {} times. Stop training.".format(num_evals))
        # This kills the outer WHILE loop.
        continue_training = False

    if batch_num % args.save_period == 0 and batch_num > 700:
      save_fname = "{0:s}/{1:s}_{2:d}.pt".format(save_model_to,
                                                 args.model_id,
                                                 batch_num)
      torch.save(
        {
          "state_dict": model.state_dict(),
          "optimizer_cls": optimizer_cls.state_dict(),
          "optimizer_enc": optimizer_enc.state_dict(),
          "scheduler_cls": scheduler_cls.state_dict()
          if scheduler_cls is not None else None,
          "scheduler_enc": scheduler_enc.state_dict()
          if scheduler_enc is not None else None,
          "args": args
        },
        save_fname
      )
      print(
        "Total {0:.2f} minutes have passed, saving at {1:s} ".format(
          (time.time() - init_time) / 60, save_fname))


"""
Test
"""

def _test(args: argparse.Namespace,
          model: Union[TransformerVecModel, TransformerBoxModel],
          device: torch.device):
  print("==> Start eval...")
  assert args.load
  save_output_to = os.path.join(constant.BASE_PATH, "outputs", args.model_id)
  if not os.path.exists(save_output_to):
    print("==> Create {}".format(save_output_to))
    os.makedirs(save_output_to, exist_ok=False)
  test_fname = args.eval_data
  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  print("==> Loading data generator... ")
  data_gens = get_datasets([(test_fname, "test")],
                           args,
                           model.transformer_tokenizer)
  print("==> Loading ID -> TYPE mapping... ")
  _word2id = constant.load_vocab_dict(constant.TYPE_FILES[args.goal])
  id2word_dict = {v: k for k, v in _word2id.items()}
  model.eval()
  load_model(args.reload_model_name,
             constant.EXP_ROOT,
             args.model_id, model)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
    print("==> use", torch.cuda.device_count(), "GPUs.")
  for name, dataset in [(test_fname, data_gens[0])]:
    print("Processing... " + name)
    total_gold_pred = []
    total_annot_ids = []
    total_probs = []
    total_ys = []
    for batch_num, batch in enumerate(dataset):
      if batch_num % 100 == 0:
        print(batch_num)
      if not isinstance(batch, dict):
        print("==> batch: ", batch)
      inputs, targets = to_torch(batch, device)
      annot_ids = batch.pop("ex_ids")
      if args.n_gpu > 1:
        output_logits = model(inputs, targets)
      else:
        _, output_logits = model(inputs)
      output_index = get_output_index(
        output_logits,
        threshold=args.threshold,
        is_prob=True if args.emb_type == "box" else False)
      if args.emb_type == "box":
        output_prob = output_logits.data.cpu().clone().numpy()
      else:
        output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = batch["targets"].data.cpu().clone()
      gold_pred = get_gold_pred_str(output_index, y, id2word_dict)
      total_probs.extend(output_prob)
      total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      total_annot_ids.extend(annot_ids)
    pickle.dump(
      {"gold_id_array": total_ys, "pred_dist": total_probs},
      open("{0:s}/pred_dist.pkl".format(save_output_to), "wb"))
    print(len(total_annot_ids), len(total_gold_pred))
    with open("{0:s}/pred_labels.json".format(save_output_to), "w") as f_out:
      output_dict = {}
      counter = 0
      for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
        output_dict[a_id] = {"gold": gold, "pred": pred}
        counter += 1
      json.dump(output_dict, f_out)
    eval_str = get_eval_string(total_gold_pred)
    print(eval_str)


def main():
  args = parser.parse_args()
  # Lower text for BERT uncased models
  args.do_lower = True if "uncased" in args.model_type else False
  # Setup CUDA, GPU & distributed training
  assert torch.cuda.is_available()
  if args.local_rank == -1:
    device = torch.device("cuda")
    args.n_gpu = 1  #torch.cuda.device_count()
  else:
    # Initializes the distributed backend which will take care of synchronizing
    # nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device
  set_seed(args.seed, args.n_gpu)
  # Load pretrained model and tokenizer
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()
  if args.emb_type == "baseline":
    print("TransformerVecModel")
    model = TransformerVecModel(args, constant.ANSWER_NUM_DICT[args.goal])
  elif args.emb_type == "box":
    print("TransformerBoxModel")
    model = TransformerBoxModel(args, constant.ANSWER_NUM_DICT[args.goal])
  else:
    raise NotImplementedError
  if args.local_rank == 0:
    torch.distributed.barrier()
  model.to(args.device)
  args.max_position_embeddings = \
      model.transformer_config.max_position_embeddings
  print("-" * 80)
  for k, v in vars(args).items():
    print(k, ":", v)
  print("-" * 80)
  if args.mode == "train":
    print("==> mode: train")
    _train(args, model, device)
  elif args.mode == "test":
    print("==> mode: test")
    _test(args, model, device)
  else:
    raise ValueError("invalid value for 'mode': {}".format(args.mode))


if __name__ == "__main__":
  main()
