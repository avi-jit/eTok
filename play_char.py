from ast import Not
from dataset import eDataset, CharDataset, eDataset_nat

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
import datasets
from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback

from mingpt.model import eGPT, eGPT_pre

import random
import collections
import pickle
import numpy as np
import math
from torch.utils.data import DataLoader

block_size = 128 # 256 # 128 # spatial extent of the model for its context
batch_size = 8 # 8 # 20
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
DATASET='mc4' #'wiki' 'shakespeare'
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
        #'xh', 'yi', 'yo', 'zh', 'zh-Latn', 'zu'
        #'ar','hi','sd'
        'sd','hi-Latn','gu'
        ]
    
if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
    text = open('/nas/home/thawani/etok/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
elif DATASET == 'wiki':
    text = ' '.join(datasets.load_dataset("wikitext", "wikitext-2-v1", split="train", )['text'])
elif DATASET == 'mc4':
    #text = ' '.join(datasets.load_dataset("mc4", languages=langs, split="train", )['text'][:100_000])
    text = ' '.join(random.sample(datasets.load_dataset("mc4", languages=langs, split="train", )['text'], 40_000))
else:
    raise NotImplementedError


model_type = 'egpt_pre' # 'egpt'
decoder = 'ar' # 'nat' 'word' 'ar'
if decoder == 'word':
    output_type = 'word'
else:
    output_type = 'nat'
if model_type == 'egpt':
    model_class = eGPT
elif model_type == 'egpt_pre':
    model_class = eGPT_pre
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

if output_type == 'word':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=27430)
elif output_type == 'nat':
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=None)
else:
    raise NotImplemented
#raise NotImplementedError
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
        n_embd=768, 
        #e2e_vocab_size=10,
        learning_rate=1e-4,
        itoc=full_dataset.itoc,
        ctoi=full_dataset.ctoi,
        itow=full_dataset.itow,
        wtoi=full_dataset.wtoi,
        num_prefix=1,
        nat_layers=2,
        bigram=full_dataset.bigram,
        decoder=decoder,
    )

    # scheduler
    lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                        final_tokens=00*len(train_set)*block_size)
    wandb_logger = pl.loggers.WandbLogger(project="etok", save_dir='/nas/ckgfs/users/thawani/etok/')
    wandb.run.name = f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {decoder} {batch_size}bs {block_size}bl {'-'.join(wandb.run.name.split('-')[:2])}"
    print(wandb.run.name)
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
                    logger=wandb_logger,
                    val_check_interval=0.25,
                    default_root_dir="/nas/ckgfs/users/thawani/etok/checkpoints/"
                    )
    #trainer.fit(model, train_loader)
    trainer.fit(model, train_loader, val_loader)
d = []
model.eval()
queries = []
with torch.no_grad():
    full_dataset.ctoi = model.config.ctoi; full_dataset.itoc = model.config.itoc

    #raise NotImplemented
    if model_type == 'egpt':
        for x,y,mask in tqdm(iter(val_loader)):
            _, attn, query = model(x, mask, eval=True) # attn is b,t,Ve. query is b,t,d.
            queries.append(query.cpu().tolist())
            #temp = attn.reshape((len(x), train_dataset.block_size, model.config.e2e_vocab_size))
            picks = torch.argmax(attn, dim=-1).tolist() # h,b,t
            for i in range(len(x)):
                sent = ' '.join([''.join([full_dataset.itoc[_] for _ in temp]).strip() for temp in x[i].tolist()]).strip()
                d.append((sent,[p[i] for p in picks]))

        a = [collections.defaultdict(lambda:[]) for _ in range(len(d[0][1]))]
        for s,x in d:
            for i,x1 in enumerate(x):
                for s1,x11 in zip(s.split(' '),x1):
                    a[i][x11].append(s1.strip())
        #print([{k:collections.Counter(v).most_common() for k,v in a1.items()} for a1 in a])

        words = collections.defaultdict(lambda:[])
        tokens = collections.defaultdict(lambda:[])
        for s,(h1,h2,h3,h4) in d:
            for _,_1,_2,_3,_4 in zip(s.split(' '),h1,h2,h3,h4):
                words[_].append((_1,_2,_3,_4))
                tokens[(_1,_2,_3,_4)].append(_)
        #if LOAD_CKPT:
        #    pickle.dump((d,[dict(a1) for a1 in a], dict(words), dict(tokens), queries), open(f"{LOAD_CKPT+'_'+DATASET}_egpt.pkl",'wb'))

        pickle.dump((d,[dict(a1) for a1 in a], dict(words), dict(tokens), queries), open(f'{DATASET}_egpt_nat.pkl','wb'))
    elif model_type == 'egpt_pre':
        ACC_C0 = []; ACC_W0 = []; ACC_C = []; BT = []
        for x,y,x_mask,y_mask in tqdm(iter(val_loader)):
            b, t, c = x.size()
            logits, query = model(x.to(DEVICE), x_mask.to(DEVICE), eval=True)
            if output_type == 'word':
                raise NotImplemented
            elif output_type == 'nat':
                acc_c0 = torch.argmax(logits,dim=-1)==y.to(DEVICE) # b,t,c
                acc_w0 = torch.all(acc_c0, dim=2); acc_w0 = acc_w0.sum()/acc_w0.numel()
                mask = y_mask.to(DEVICE).view(-1); mask = torch.arange(c, device=mask.device).expand(len(mask), c) < mask.unsqueeze(1); mask = mask.view(b,t,c)
                acc_c = (acc_c0 * mask).sum()/mask.sum(); acc_c0 = acc_c0.sum()/acc_c0.numel()
                ACC_C.append(acc_c.item()); ACC_C0.append(acc_c0.item()); ACC_W0.append(acc_w0.item()); BT.append(b*t)
            else:
                raise NotImplemented
        #print(collections.Counter(y.reshape(-1).cpu().tolist()).most_common(5))
        #print(collections.Counter(torch.argmax(logits,dim=-1).reshape(-1).cpu().tolist()).most_common(5))
print("dumped")

# alright, let's sample some character-level shakespear
#from mingpt.utils import sample

#context = "O God, I code but"
#x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
#y = sample(model, x, 1000, temperature=0.9, sample=True, top_k=5)[0]
#completion = ''.join([train_dataset.itos[int(i)] for i in y])
#print(completion)

