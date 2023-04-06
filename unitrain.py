from ast import Not
from dataset import myDataset
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

#from mingpt.model import eGPT, eGPT_pre, ByT5
from newmodel import myGPT

import collections
import pickle
import numpy as np
import math
from torch.utils.data import DataLoader
import random
import argparse

def main(
        DATASET='trial', 
        DEVICE=0, 
        NUM_PREFIX=4, 
        #block_size=128, 
        block_size=256,
        batch_size=8, 
        base='word', 
        do_e2e=False,
        EPOCHS=400,
        LOAD_CKPT=None,
        debug=False,
    ):
    
    if LOAD_CKPT:
        model = myGPT.load_from_checkpoint(LOAD_CKPT, 
            #block_size=32
        )
        block_size = model.config.block_size
        model.to(DEVICE)
        base = model.config.base
        NUM_PREFIX = model.config.num_prefix
        do_e2e = (NUM_PREFIX != 0)
        #DATASET = model.config.dataset
        print(f"loaded: {base=} {NUM_PREFIX=} {do_e2e=}")
        vocab = model.config.vocab
        #maxlen = model.config.maxlen
        
    CACHE_DIR="/home1/xzhu9839/data" # default ~/.cache/huggingface/datasets
    if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
        text = open('/home1/xzhu9839/data/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
    elif DATASET == 'wiki':
        text = ' '.join(datasets.load_dataset("wikitext", "wikitext-2-v1", split="train", cache_dir=CACHE_DIR)['text'])
    elif DATASET == 'mc4':
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
        #text = ' '.join(datasets.load_dataset("mc4", languages=langs, split="train", )['text'][:100_000])
        text = ' '.join(random.sample(datasets.load_dataset("mc4", languages=langs, split="train", cache_dir=CACHE_DIR)['text'], 40_000))
    elif DATASET == 'trial':
        text = "the quick brown fox jumps over the lazy dog "*1000
    elif DATASET == 'indic-ta': # 21,546,553 characters, 974 unique; 9,463,844 words, 973 unique. maxlen 3
        if False:
            lang = 'indic-ta'
            ds = []
            for i,row in tqdm(enumerate(datasets.load_dataset(f"bigscience-data/roots_{lang}_wikipedia", 
                split="train", streaming=True, use_auth_token=True).shuffle(seed=42, buffer_size=1_000))):
                if i == 10_000:
                    break
                ds.append(row['text'])
            #pickle.dump(' '.join(ds), open(f"roots_{lang}_wiki.pkl", 'wb')) 92040 characters, 101 unique; 11131 words, 1295 unique
        text = pickle.load(open('/nas/home/thawani/etok/roots_indic-ta_wiki.pkl','rb'))
    elif DATASET == 'indic-hi': # data has 16,666,528 characters, 1725 unique; 3137071 words, 209,167 unique. maxlen 185
        text = pickle.load(open('/nas/home/thawani/etok/roots_indic-hi_wiki.pkl','rb'))
    else:
        print(f"unknown dataset: {DATASET}")
        return
        #raise NotImplementedError
    if debug:
        text = text[:10_000]
    if LOAD_CKPT:
        full_dataset = myDataset(text, block_size=block_size, base=base, do_e2e=do_e2e, vocab=vocab,)
    else:
        full_dataset = myDataset(text, block_size=block_size, base=base, do_e2e=do_e2e)
    print('dataset initialized')
    # use 20% of training data for validation
    train_set_size = int(len(full_dataset) * 0.8)
    valid_set_size = len(full_dataset) - train_set_size

    # split the train set into two
    #seed = torch.Generator().manual_seed(42)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_set_size, valid_set_size])
    print('train set: ', train_set_size)
    print('val set: ', valid_set_size)
    #train_loader = DataLoader(train_dataset, batch_size=20, num_workers=16)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

    '''neptune_logger = NeptuneLogger(
        project="jaunts/etok",
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        log_model_checkpoints=False,
        #name=f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl"
    )'''   
    logger = pl.loggers.WandbLogger(project="etok", save_dir='/home1/xzhu9839/data/etok/')
    #wandb.run.name = f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl {'-'.join(wandb.run.name.split('-')[:2])}"
    wandb.run.name = f"{'debug_' if debug else ''}{NUM_PREFIX if do_e2e else ''}_{base}_{DATASET}_{logger.experiment.name}_{EPOCHS}ep"
    #logger.log_hyperparams(params=model.config)
    # logger = CometLogger(api_key=os.environ["COMET_API_KEY"],project_name="etok")
    # logger.experiment.set_name(f"{'debug_' if debug else ''}{NUM_PREFIX if do_e2e else ''}_{base}_{DATASET}_{logger.experiment.name}_{EPOCHS}ep")
    # logger.experiment.log_parameter("dataset", DATASET)
    # with logger.experiment.train():
    #     logger.experiment.log_parameter("size", train_set_size)
    # with logger.experiment.validate():
    #     logger.experiment.log_parameter("size", valid_set_size)
        
    if not full_dataset.do_e2e:
        NUM_PREFIX=0
    if not LOAD_CKPT:
        model = myGPT(
            #in_vocab_size=full_dataset.in_vocab_size,
            #out_vocab_size=full_dataset.out_vocab_size,
            block_size=full_dataset.block_size,
            n_layer=8, 
            n_head=8, 
            n_embd=512, 
            #e2e_vocab_size=10,
            learning_rate=1e-4,
            vocab=full_dataset.vocab,
            n_e2e_layer=2,
            base=full_dataset.base,
            num_prefix=NUM_PREFIX,
            canvas_size=full_dataset.maxlen,
        )
        print('model initialized')

        # scheduler
        lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                            final_tokens=00*len(train_set)*block_size)
        
        trainer = Trainer(
			accelerator="gpu",
                        profiler="simple",
                        devices=[DEVICE], 
                        precision=16, 
                        max_epochs=EPOCHS,
                        gradient_clip_val=1.0, 
                        callbacks=[lr_decay], 
                        #progress_bar_refresh_rate=1, 
                        #row_log_interval=1,
                        #log_every_n_steps=15,
                        logger=logger,
                        check_val_every_n_epoch=3,
                        #val_check_interval=1,
                        default_root_dir="/home1/xzhu9839/data/checkpoints/",
                        )
        print('trainer initialized')
        #trainer.fit(model, train_loader)
        #model.hparams.itoc = None
        #trainer.model.hparams.values()
        print('training start')
        trainer.fit(model, train_loader, val_loader)
    else:
        lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                            final_tokens=00*len(train_set)*block_size)   
        #logger.experiment.set_name(f"eval_{logger.experiment.name}")
        trainer = Trainer(
			    accelerator="gpu",
                            profiler="simple",
                            devices=[DEVICE], 
                            precision=16, 
                            max_epochs=EPOCHS,
                            gradient_clip_val=1.0, 
                            callbacks=[lr_decay],
                            logger=logger,
                            check_val_every_n_epoch=3,
                            #val_check_interval=1,
                            default_root_dir="/home1/xzhu9839/data/checkpoints/",
                            )               
        print('training start')
        trainer.fit(model, train_loader, val_loader)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-base", type=str, default='sub') # byte, char, sub, word
    parser.add_argument("--ckpt", default=False) # 
    parser.add_argument("-dataset", type=str, default='shakespeare') # shakespeare, mc4, trial
    parser.add_argument("--num_prefix", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=128+64)
    parser.add_argument("--batch_size", type=int, default=2)
    #parser.add_argument("-e2e", type=bool, default=True)
    parser.add_argument('--e2e', default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.ckpt == False:
        main(DATASET=args.dataset, DEVICE=args.device, NUM_PREFIX=args.num_prefix, base=args.base, do_e2e=args.e2e, EPOCHS=args.num_epochs, 
             block_size=args.block_size, batch_size=args.batch_size, debug=False)
    else:
        main(LOAD_CKPT=f"/home1/xzhu9839/data/etok/etok/efgqdoez/checkpoints/epoch=398-step=407379.ckpt", DEVICE=args.device,  NUM_PREFIX=args.num_prefix, base=args.base, do_e2e=args.e2e, EPOCHS=args.num_epochs,
             block_size=args.block_size, batch_size=args.batch_size, debug=False, DATASET="shakespeare") # word
# nohup python unitrain.py -dataset shakespeare -base sub --no-e2e --device 2 &
