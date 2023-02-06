from ast import Not
from dataset import eDataset, CharDataset, eDataset_nat

import torch
import torch.nn as nn
from torch.nn import functional as F
# make deterministic
import pytorch_lightning as pl

#from minGPT.play_char import LOAD_CKPT
#from pytorch_lightning import seed_everything
pl.seed_everything(42)
import regex as re
from tqdm import tqdm 
import wandb
import datasets
from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback

from mingpt.model import eGPT, eGPT_pre

import collections
import pickle
import numpy as np
import math
from torch.utils.data import DataLoader

block_size = 128 # 256 # 128 # spatial extent of the model for its context
batch_size = 8 # 8 # 20
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
DATASET='wiki' # 'shakespeare'
DEVICE=2

if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
    text = open('/nas/home/thawani/etok/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
elif DATASET == 'wiki':
    text = ' '.join(datasets.load_dataset("wikitext", "wikitext-2-v1", split="train", )['text'])
else:
    raise NotImplementedError


model_type = 'egpt_pre' # 'egpt'
output_type = 'nat' # 'word' # 'nat'
if model_type == 'egpt':
    model_class = eGPT
elif model_type == 'egpt_pre':
    model_class = eGPT_pre
else:
    raise NotImplementedError
#LOAD_CKPT="~/nas/ckgfs/users/thawani/etok/checkpoints/210v9lbr/checkpoints/epoch=49-step=76947.ckpt" # word 8bs 128bl 50ep splendid jazz 210v9lbr
#LOAD_CKPT="~/nas/ckgfs/users/thawani/etok/checkpoints/epoch=9-step=15387.ckpt" # word 8bs 128bl 10ep splendid jazz 3p9xpv7p
LOAD_CKPT="/nas/ckgfs/users/thawani/etok/checkpoints/14gn9wns/checkpoints/epoch=29-step=45015.ckpt"
#LOAD_CKPT="/nas/ckgfs/users/thawani/etok/etok/3vc0gj1c/checkpoints/epoch=49-step=76947.ckpt"

model = model_class.load_from_checkpoint(LOAD_CKPT, 
    #block_size=32
)
block_size = model.block_size
word_vocab_size = model.config.out_vocab_size
model.to(DEVICE)

if output_type == 'word':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=word_vocab_size)
elif output_type == 'nat':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=None)
else:
    raise NotImplemented
# use 20% of training data for validation
train_set_size = int(len(full_dataset) * 0.8)
valid_set_size = len(full_dataset) - train_set_size

# split the train set into two
#seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_set_size, valid_set_size])

#train_loader = DataLoader(train_dataset, batch_size=20, num_workers=16)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

# scheduler
lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                    final_tokens=00*len(train_set)*block_size)
wandb_logger = pl.loggers.WandbLogger(project="etok", save_dir='/nas/ckgfs/users/thawani/etok/')
wandb.run.name = f"{DATASET} {model_type}{model.config.num_prefix} {output_type} {batch_size}bs {block_size}bl {'-'.join(wandb.run.name.split('-')[:2])}"

#assert full_dataset.wtoi == model.config.wtoi
full_dataset.wtoi = model.config.wtoi
full_dataset.itow = model.config.itow
full_dataset.ctoi = model.config.ctoi
full_dataset.itoc = model.config.itoc

trainer = Trainer(#accelerator="cpu",
            profiler="simple",
            accelerator="gpu", devices=[DEVICE], 
            #precision=16, 
            max_epochs=100,
            gradient_clip_val=1.0, 
            #callbacks=[lr_decay], 
            #progress_bar_refresh_rate=1, 
            #row_log_interval=1,
            #log_every_n_steps=15,
            logger=wandb_logger,
            val_check_interval=0.25,
            default_root_dir="/nas/ckgfs/users/thawani/etok/checkpoints/",
            #ckpt_path=LOAD_CKPT, # new only
            resume_from_checkpoint=LOAD_CKPT,
            )
#trainer.fit(model, train_loader)
trainer.fit(model, train_loader, val_loader)