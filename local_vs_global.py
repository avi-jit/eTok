from ast import Not
from dataset import eDataset, CharDataset, eDataset_nat, eDataset_char, eDataset_sub

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
# make deterministic
import pytorch_lightning as pl
#from pytorch_lightning import seed_everything
pl.seed_everything(42)
import regex as re
from tqdm import tqdm 
import wandb
#import neptune.new as neptune

import datasets
from pytorch_lightning import Trainer
#from pytorch_lightning.loggers import NeptuneLogger
import comet_ml
from pytorch_lightning.loggers import CometLogger
from mingpt.lr_decay import LearningRateDecayCallback

from mingpt.model import eGPT, eGPT_pre, ByT5

import collections
import pickle
import numpy as np
import math
from torch.utils.data import DataLoader
import random

block_size = 128 * 5 # 256 # 128 # spatial extent of the model for its context
batch_size = 8 # 8 # 20
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
DATASET='mc4' 
#DATASET='shakespeare'
DEVICE=0
if DATASET != 'mc4':
    langs = ['en']
else:
    langs = [
        #'af', 'am', 'ar', 'az', 'be', 'bg', 'bg-Latn', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'el-Latn', 'en', 
        # 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hi-Latn', 'hmn', 'ht', 'hu', 
        # 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'ja-Latn', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 
        # 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'ru-Latn', 'sd', 
        # 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tr', 'uk', 'und', 'ur', 'uz', 'vi', 
        #'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu' # these were first few experiments when langs not logged in wandb
        #'ar','hi','sd',
        'sd','hi-Latn','gu',
        ]
CACHE_DIR="/nas/ckgfs/users/thawani/hf_cache/datasets" # default ~/.cache/huggingface/datasets
if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
    text = open('/nas/home/thawani/etok/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
elif DATASET == 'wiki':
    text = ' '.join(datasets.load_dataset("wikitext", "wikitext-2-v1", split="train", cache_dir=CACHE_DIR)['text'])
elif DATASET == 'mc4':
    #text = ' '.join(datasets.load_dataset("mc4", languages=langs, split="train", )['text'][:100_000])
    text = ' '.join(random.sample(datasets.load_dataset("mc4", languages=langs, split="train", cache_dir=CACHE_DIR)['text'], 40_000))
else:
    raise NotImplementedError

base = 'sub' # 'word'? 'char' 'byte' 'sub'
model_type = 'byt5' # 'egpt_pre' = prefix + word (output type word or nat); 'byt5' = no bottleneck
output_type = 'nat' # 'nat' 'word'
if model_type == 'egpt':
    model_class = eGPT
elif model_type == 'egpt_pre':
    model_class = eGPT_pre
elif model_type == 'byt5':
    model_class = ByT5
else:
    raise NotImplementedError
LOAD_CKPT=None
#LOAD_CKPT="etok/2oqlujjk/checkpoints/epoch=6-step=168448.ckpt"
#LOAD_CKPT="~/etok/minGPT/etok/2ydwnrrq/checkpoints/epoch=49-step=32750.ckpt"
#LOAD_CKPT="~/etok/etok/2hdg50sk/checkpoints/epoch=34-step=22925.ckpt"
#LOAD_CKPT="~/etok/etok/2ickyuc4/checkpoints/epoch=7-step=24440.ckpt"
#LOAD_CKPT="~/etok/etok/1dc71nis/checkpoints/epoch=7-step=24440.ckpt"
if LOAD_CKPT:
    model = model_class.load_from_checkpoint(LOAD_CKPT, 
    #block_size=32
    )
    block_size = model.block_size
    model.to(DEVICE)
if model_type == 'byt5':
    if base == 'char':
        full_dataset = eDataset_char(text, block_size)
    elif base == 'sub':
        full_dataset = eDataset_sub(text, block_size) 
    full_dataset.bigram = None
elif output_type == 'word':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=27430)
elif output_type == 'nat':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=None)
else:
    raise NotImplementedError
# use 20% of training data for validation
train_set_size = int(len(full_dataset) * 0.8)
valid_set_size = len(full_dataset) - train_set_size

# split the train set into two
#seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_set_size, valid_set_size])

#train_loader = DataLoader(train_dataset, batch_size=20, num_workers=16)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

if not LOAD_CKPT:
    if output_type == 'word':
        out_vocab_size=full_dataset.wvocab_size
    else:
        out_vocab_size=None
    model = model_class(
        vocab_size=full_dataset.cvocab_size,
        out_vocab_size=out_vocab_size,
        block_size=full_dataset.block_size,
        n_layer=8, 
        n_head=8, 
        n_embd=512, 
        #e2e_vocab_size=10,
        learning_rate=1e-4,
        itoc=full_dataset.itoc,
        ctoi=full_dataset.ctoi,
        itow=full_dataset.itow,
        wtoi=full_dataset.wtoi,
        num_prefix=4,
        nat_layers=2,
        bigram=full_dataset.bigram,
        base=base,
    )

    # scheduler
    lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                        final_tokens=00*len(train_set)*block_size)

    '''neptune_logger = NeptuneLogger(
        project="jaunts/etok",
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        log_model_checkpoints=False,
        #name=f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl"
    )'''   
    #wandb_logger = pl.loggers.WandbLogger(project="etok", save_dir='/nas/ckgfs/users/thawani/etok/')
    #wandb.run.name = f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl {'-'.join(wandb.run.name.split('-')[:2])}"
    #neptune_logger.log_hyperparams(params=model.config)
    
    trainer = Trainer(#accelerator="cpu",
                    profiler="simple",
                    accelerator="gpu", devices=[DEVICE], 
                    #precision=16, 
                    max_epochs=50,
                    gradient_clip_val=1.0, 
                    callbacks=[lr_decay], 
                    #progress_bar_refresh_rate=1, 
                    #row_log_interval=1,
                    #log_every_n_steps=15,
                    #logger=wandb_logger,
                    #logger=neptune_logger,
                    logger=CometLogger(api_key=os.environ["COMET_API_KEY"],project_name="etok"),
                    val_check_interval=0.25,
                    default_root_dir="/nas/ckgfs/users/thawani/etok/checkpoints/"
                    )
    #trainer.fit(model, train_loader)
    #model.hparams.itoc = None
    #trainer.model.hparams.values()
    trainer.fit(model, train_loader, val_loader)
d = []
model.eval()