# Modeling Fine-Grained Entity Types with Box Embeddings

> [**Modeling Fine-Grained Entity Types with Box Embeddings**](https://arxiv.org/pdf/2101.00345.pdf)<br/>
> Yasumasa Onoe, Michael Boratko, Andrew McCallum, Greg Durrett<br/>
> ACL 2021

```bibtex
@inproceedings{onoe2021boxet,
 title={Modeling Fine-Grained Entity Types with Box Embeddings},
 author={Yasumasa Onoe, Michael Boratko, Andrew McCallum, Greg Durrett},
 booktitle={ACL},
 year={2021}
}
```

## Getting Started 

### Dependencies

```bash
$ git clone https://github.com/yasumasaonoe/Box4Types.git
```

This code has been tested with Python 3.7 and the following dependencies:

- `torch==1.7.1` (Please install the right version of Pytorch depending on your CUDA version.)
- `transformers==4.9.2`
- `wandb==0.12.1`

If you're using a conda environment, please use the following commands:

```bash
$ conda create -n box4et python=3.7
$ conda activate box4et
$ pip install  [package name]
```

### File Descriptions

- `box4et/main.py`: Main script for training and evaluating models, and writing predictions to an output file.
- `box4et/models.py`: Defines a Transformer-based entity typing model.
- `box4et/data_utils.py`: Contains data loader and utility functions.
- `box4et/constant.py`: Defines paths etc.
- `box4et/scorer.py`: Compute precision, recall, and F1 given an output file.
- `box4et/train_*.sh`: Sample training command.
- `box4et/eval_*.sh`: Sample evaluation command.

## Datasets / Models

This code assumes 3 directories listed below. Paths to these directories are specified in `box4et/constant.py`.
- `./data`: This directory contains train/dev data files.
- `./data/ontology`: This directory contains type vocab files. 
- `./model`: Trained models will be saved in this directory. When you run `main.py` with the test mode, the trained model is loaded from here.
- Download model checkpoints (box and vector models for 4 datasets) from [here](https://drive.google.com/file/d/1Rt5M_9MC7x1C7J-c_QhNjXC22b9Y1e_U/view?usp=sharing) (NOTE: total size is around 30GB). 
- UFET: We do not include the augmented UFET training set since it is derived from English Gigaword, which belongs to LDC. If you have a LDC membership and want to use the augmented data, please contact at <yasumasa@utexas.edu>.

Run this to download these folders.
```bash
$ bash download_data.sh
```




The data files are formatted as jsonlines. Here is an example from UFET:
```
{
    "ex_id": "dev_190", 
    "right_context": ["."], 
    "left_context": ["For", "this", "handpicked", "group", "of", "jewelry", "savvy", "Etsy", "artisans", ",", "their", "passion", "is", "The", "Hunger", "Games", ",", "the", "first", "of", "3", "best", "selling", "young", "adult", "books", "by"], 
    "right_context_text": ".", 
    "left_context_text": "For this handpicked group of jewelry savvy Etsy artisans , their passion is The Hunger Games , the first of 3 best selling young adult books by",
    "y_category": ["name", "person", "writer", "author"],
    "word": "Suzanne Collins", 
    "mention_as_list": ["Suzanne", "Collins"]
}

```

| Field                     | Description                                                                              |
|---------------------------|------------------------------------------------------------------------------------------|
| `ex_id`                   | Unique example ID.                                                                       |
| `right_context`           | Tokenized right context of a mention.                                                    |
| `left_context`            | Tokenized left context of a mention.                                                     |
| `word`                    | A mention.                                                                               |
| `right_context_text`      | Right context of a mention.                                                              |
| `left_context_text`       | Left context of a mention.                                                               |
| `y_category`              | The gold entity types derived from Wikipedia categories.                                 |
| `y_title`                 | Wikipedia title of the gold Wiki entity.                                                 |
| `mention_as_list`         | A tokenized mention.                                                                     |


## Entity Typing Training and Evaluation

### Training

`main.py` is the primary script for training and evaluating models. See `box4et/train_*.sh`.

```bash
$ cd box4et
$ bash train_box.sh
```

### Evaluation

If you would like to evaluate the trained model on another dataset, simply set `--mode` to `test` and point to the test data using `--eval_data`. Make sure put `-load` so that the trained model will be loaded. See `box4et/eval_*.sh`.

```bash
$ cd box4et
$ bash eval_box.sh
```
