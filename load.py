from ast import Not
from dataset import eDataset, CharDataset, eDataset_nat

import torch
import torch.nn as nn
from torch.nn import functional as F

# make deterministic
import pytorch_lightning as pl

# from minGPT.play_char import LOAD_CKPT
# from pytorch_lightning import seed_everything
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

block_size = 128  # 256 # 128 # spatial extent of the model for its context
batch_size = 8  # 8 # 20
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
DATASET = "wiki"  # 'shakespeare'
DEVICE = 1

if DATASET == "shakespeare":  # one line of poem is roughly 50 characters
    text = open(
        "/nas/home/thawani/etok/tinyshake.txt", "r"
    ).read()  # don't worry we won't run out of file handles
elif DATASET == "wiki":
    text = " ".join(
        datasets.load_dataset(
            "wikitext",
            "wikitext-2-v1",
            split="train",
        )["text"]
    )
else:
    raise NotImplementedError


model_type = "egpt_pre"  # 'egpt'
output_type = "nat"  # 'nat'
if model_type == "egpt":
    model_class = eGPT
elif model_type == "egpt_pre":
    model_class = eGPT_pre
else:
    raise NotImplementedError
# LOAD_CKPT=None
# LOAD_CKPT="etok/2oqlujjk/checkpoints/epoch=6-step=168448.ckpt"
# LOAD_CKPT="~/etok/minGPT/etok/2ydwnrrq/checkpoints/epoch=49-step=32750.ckpt"
# LOAD_CKPT="~/etok/etok/2hdg50sk/checkpoints/epoch=34-step=22925.ckpt"
# LOAD_CKPT="~/etok/etok/2ickyuc4/checkpoints/epoch=7-step=24440.ckpt"
# LOAD_CKPT="~/etok/etok/1dc71nis/checkpoints/epoch=7-step=24440.ckpt"
# LOAD_CKPT="~/etok/etok/3kd2ez5i/checkpoints/epoch=9-step=30900.ckpt"
# LOAD_CKPT="~/nas/ckgfs/users/thawani/etok/checkpoints/epoch=29-step=45015.ckpt" # nat 8bs 128bl 30ep magic shadow 14gn9wns - uploaded tsv to drive
# LOAD_CKPT="/nas/ckgfs/users/thawani/etok/checkpoints/1nzbchk3/checkpoints/epoch=9-step=14619.ckpt" # nat 8bs 128bl 10ep ethereal bee 1nzbchk3
LOAD_CKPT = "/nas/ckgfs/users/thawani/etok/etok/3ddc3o9y/checkpoints/epoch=15-step=23469.ckpt"  # nat 8bs 128bl 50ep comfy resonance 3ddc3o9y 0.1 bigram mixing
# LOAD_CKPT="~/nas/ckgfs/users/thawani/etok/checkpoints/210v9lbr/checkpoints/epoch=49-step=76947.ckpt" # word 8bs 128bl 50ep splendid jazz 210v9lbr
# LOAD_CKPT="~/nas/ckgfs/users/thawani/etok/checkpoints/epoch=9-step=15387.ckpt" # word 8bs 128bl 10ep splendid jazz 3p9xpv7p

model = model_class.load_from_checkpoint(
    LOAD_CKPT,
    # block_size=32
)
block_size = model.block_size
word_vocab_size = model.config.out_vocab_size
model.to(DEVICE)

if output_type == "word":
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=word_vocab_size)
elif output_type == "nat":
    full_dataset = eDataset_nat(text, block_size, word_vocab_size=None)
else:
    raise NotImplemented

model.bigram = torch.ones(model.config.vocab_size + 1, model.config.vocab_size)
for (i, j), f in full_dataset.bigram.items():
    model.bigram[model.config.ctoi.get(i, 281), model.config.ctoi.get(j)] += f

# use 20% of training data for validation
train_set_size = int(len(full_dataset) * 0.8)
valid_set_size = len(full_dataset) - train_set_size

# split the train set into two
# seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(
    full_dataset, [train_set_size, valid_set_size]
)

# train_loader = DataLoader(train_dataset, batch_size=20, num_workers=16)
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

d = []
model.eval()
queries = []
with torch.no_grad():
    full_dataset.ctoi = model.config.ctoi
    full_dataset.itoc = model.config.itoc
    # full_dataset.wtoi = model.config.wtoi; full_dataset.itow = model.config.itow

    ACC_W0 = []
    BT = []
    if model_type == "egpt":
        for x, y, mask in tqdm(iter(val_loader)):
            _, attn, query = model(
                x, mask, eval=True
            )  # attn is b,t,Ve. query is b,t,d.
            queries.append(query.cpu().tolist())
            # temp = attn.reshape((len(x), train_dataset.block_size, model.config.e2e_vocab_size))
            picks = torch.argmax(attn, dim=-1).tolist()  # h,b,t
            for i in range(len(x)):
                sent = " ".join(
                    [
                        "".join([full_dataset.itoc[_] for _ in temp]).strip()
                        for temp in x[i].tolist()
                    ]
                ).strip()
                d.append((sent, [p[i] for p in picks]))

        a = [collections.defaultdict(lambda: []) for _ in range(len(d[0][1]))]
        for s, x in d:
            for i, x1 in enumerate(x):
                for s1, x11 in zip(s.split(" "), x1):
                    a[i][x11].append(s1.strip())
        # print([{k:collections.Counter(v).most_common() for k,v in a1.items()} for a1 in a])

        words = collections.defaultdict(lambda: [])
        tokens = collections.defaultdict(lambda: [])
        for s, (h1, h2, h3, h4) in d:
            for _, _1, _2, _3, _4 in zip(s.split(" "), h1, h2, h3, h4):
                words[_].append((_1, _2, _3, _4))
                tokens[(_1, _2, _3, _4)].append(_)
        # if LOAD_CKPT:
        #    pickle.dump((d,[dict(a1) for a1 in a], dict(words), dict(tokens), queries), open(f"{LOAD_CKPT+'_'+DATASET}_egpt.pkl",'wb'))

        pickle.dump(
            (d, [dict(a1) for a1 in a], dict(words), dict(tokens), queries),
            open(f"{DATASET}_egpt_nat.pkl", "wb"),
        )
    elif model_type == "egpt_pre":
        if output_type == "nat":
            ACC_C0 = []
            ACC_C = []
            for x, y, x_mask, y_mask in tqdm(iter(val_loader)):
                b, t, c = x.size()
                logits, query = model(x.to(DEVICE), x_mask.to(DEVICE), eval=True)
                acc_c0 = torch.argmax(logits, dim=-1) == y.to(DEVICE)  # b,t,c
                acc_w0 = torch.all(acc_c0, dim=2)
                acc_w0 = acc_w0.sum() / acc_w0.numel()
                mask = y_mask.to(DEVICE).view(-1)
                mask = torch.arange(c, device=mask.device).expand(
                    len(mask), c
                ) < mask.unsqueeze(1)
                mask = mask.view(b, t, c)
                acc_c = (acc_c0 * mask).sum() / mask.sum()
                acc_c0 = acc_c0.sum() / acc_c0.numel()
                ACC_C.append(acc_c.item())
                ACC_C0.append(acc_c0.item())
                ACC_W0.append(acc_w0.item())
                BT.append(b * t)
        elif output_type == "word":
            for x, y, x_mask in tqdm(iter(val_loader)):
                b, t, c = x.size()
                logits, query = model(x.to(DEVICE), x_mask.to(DEVICE), eval=True)
                preds = torch.argmax(logits, dim=-1)
                acc_w0 = (preds == y.to(DEVICE)).float().mean()
                BT.append(b * t)
                ACC_W0.append(acc_w0.item())
        else:
            raise NotImplemented
        # print(collections.Counter(y.reshape(-1).cpu().tolist()).most_common(5))
        # print(collections.Counter(torch.argmax(logits,dim=-1).reshape(-1).cpu().tolist()).most_common(5))
print("dumped")

# alright, let's sample some character-level shakespear
# from mingpt.utils import sample

# context = "O God, I code but"
# x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
# y = sample(model, x, 1000, temperature=0.9, sample=True, top_k=5)[0]
# completion = ''.join([train_dataset.itos[int(i)] for i in y])
# print(completion)
