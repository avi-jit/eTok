from ast import Not
from dataset import myDataset
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# make deterministic
import pytorch_lightning as pl

# from pytorch_lightning import seed_everything
pl.seed_everything(42)
import regex as re
from tqdm import tqdm
import wandb
from collections import Counter

# import neptune.new as neptune

import datasets
from pytorch_lightning import Trainer

# from pytorch_lightning.loggers import NeptuneLogger
# import comet_ml
# from pytorch_lightning.loggers import CometLogger
from mingpt.lr_decay import LearningRateDecayCallback

# from mingpt.model import eGPT, eGPT_pre, ByT5
from newmodel import myGPT

import collections
import pickle
import numpy as np
import math
from torch.utils.data import DataLoader
import random

# random.seed(42)
import argparse

torch.cuda.empty_cache()


def main(
    DATASET="trial",
    DEVICE=0,
    NUM_PREFIX=4,
    # block_size=128,
    block_size=256,
    batch_size=8,
    base="word",
    do_e2e=False,
    report_numeracy=False,
    EPOCHS=1,
    LOAD_CKPT=None,
    debug=False,
    LANG="en",
    learning_rate=1e-4,
    USE_LOGGER="true",
    fixed_context_char_size=150,
):
    if LOAD_CKPT:
        model = myGPT.load_from_checkpoint(
            LOAD_CKPT,
            batch_size=batch_size,
            lang=LANG,
            dataset=DATASET,
            save_to_val_csv=True,
            fixed_context_char_size=fixed_context_char_size,
            report_numeracy=report_numeracy,
        )
        block_size = model.config.block_size
        model.to(DEVICE)
        base = model.config.base
        NUM_PREFIX = model.config.num_prefix
        do_e2e = NUM_PREFIX != 0
        # DATASET = model.config.dataset
        print(f"loaded: {base=} {NUM_PREFIX=} {do_e2e=}")
        vocab = model.config.vocab
        # maxlen = model.config.maxlen

    CACHE_DIR = "/scratch1/sghaneka/datasets"  # default ~/.cache/huggingface/datasets
    if DATASET == "shakespeare":  # one line of poem is roughly 50 characters
        text = open(
            "/scratch1/sghaneka/etok/tinyshake.txt", "r"
        ).read()  # don't worry we won't run out of file handles
    elif DATASET == "custom":
        text = open(f"/home1/sghaneka/datasets_for_etok/{LANG}_10000.txt", "r").read()
    elif DATASET == "wiki-convert":
        text = open("/home1/sghaneka/datasets_for_etok/wiki_convert.txt", "r").read()
    elif DATASET == "trial":
        text = "the quick brown fox jumps over the lazy dog " * 1000
    else:
        print(f"unknown dataset: {DATASET}")
        return
        # raise NotImplementedError
    if debug:
        text = text[:10_000]
    if LOAD_CKPT:
        full_dataset = myDataset(
            text,
            block_size=block_size,
            base=base,
            do_e2e=do_e2e,
            vocab=vocab,
        )

        words = re.findall(r"\w+", text)
        frequency_groups = collections.defaultdict(list)
        for word, freq in Counter(words).most_common():
            if freq > 45:
                frequency_groups[1].append(word)
            elif 45 > freq > 10:
                frequency_groups[2].append(word)
            else:
                frequency_groups[3].append(word)

        text_for_metric_generation = [len(val) for _, val in frequency_groups.items()]
        # frequency group 1 -> greater than 45
        # frequency group 2 -> less than 45 greater than 10
        # frequency group 3 -> less than 10

    else:
        full_dataset = myDataset(text, block_size=block_size, base=base, do_e2e=do_e2e)

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

    if USE_LOGGER == "true":
        """neptune_logger = NeptuneLogger(
            project="jaunts/etok",
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            log_model_checkpoints=False,
            #name=f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl"
        )"""
        logger = pl.loggers.WandbLogger(
            project="etok",
            save_dir="/scratch1/sghaneka/etok/",
            settings=wandb.Settings(start_method="fork"),
        )
        # wandb.run.name = f"{DATASET} {'-'.join(langs)} {model_type}{model.config.num_prefix} {base} {output_type} {batch_size}bs {block_size}bl {'-'.join(wandb.run.name.split('-')[:2])}"
        wandb.run.name = f"{'debug_' if debug else ''}{NUM_PREFIX if do_e2e else 'no-e2e'}_{base}_{DATASET}_{LANG}_{logger.experiment.name}_{EPOCHS}ep"
        # logger = CometLogger(api_key=os.environ["COMET_API_KEY"],project_name="etok")
        # logger.experiment.set_name(f"{'debug_' if debug else ''}{NUM_PREFIX if do_e2e else ''}_{base}_{DATASET}_{logger.experiment.name}_{EPOCHS}ep")
        # logger.experiment.log_parameter("dataset", DATASET)
        # with logger.experiment.train():
        #     logger.experiment.log_parameter("size", train_set_size)
        # with logger.experiment.validate():
        #     logger.experiment.log_parameter("size", valid_set_size)

    if not full_dataset.do_e2e:
        NUM_PREFIX = 0
    if not LOAD_CKPT:
        model = myGPT(
            # in_vocab_size=full_dataset.in_vocab_size,
            # out_vocab_size=full_dataset.out_vocab_size,
            block_size=full_dataset.block_size,
            n_layer=8,
            n_head=8,
            n_embd=512,
            # e2e_vocab_size=10,
            learning_rate=learning_rate,
            vocab=full_dataset.vocab,
            n_e2e_layer=2,
            base=full_dataset.base,
            num_prefix=NUM_PREFIX,
            canvas_size=full_dataset.maxlen,
            fixed_context_char_size=fixed_context_char_size,
            report_numeracy=report_numeracy,
        )

        logger.log_hyperparams(params=model.config)

        # scheduler
        lr_decay = LearningRateDecayCallback(
            learning_rate=learning_rate,
            warmup_tokens=512 * 20,
            final_tokens=00 * len(train_set) * block_size,
        )

        trainer = Trainer(  # accelerator="cpu",
            profiler="simple",
            accelerator="gpu",
            devices=[DEVICE],
            precision=16,
            max_epochs=EPOCHS,
            gradient_clip_val=1.0,
            callbacks=[lr_decay],
            # progress_bar_refresh_rate=1,
            # row_log_interval=1,
            # log_every_n_steps=15,
            logger=logger,
            # val_check_interval=0.9,
            check_val_every_n_epoch=5,
            default_root_dir="/scratch1/sghaneka/etok/checkpoints/",
        )
        # trainer.fit(model, train_loader)
        # model.hparams.itoc = None
        # trainer.model.hparams.values()
        trainer.fit(model, train_loader, val_loader)
    else:
        model.eval()
        # logger.experiment.set_name(f"eval_{logger.experiment.name}")
        with torch.no_grad():
            trainer = Trainer(  # accelerator="cpu",
                # profiler="simple",
                accelerator="gpu",
                devices=[DEVICE],
                # precision=16,
                max_epochs=1,
                gradient_clip_val=1.0,
                # logger=logger,
                val_check_interval=0.5,
                default_root_dir="/scratch1/sghaneka/etok/checkpoints/",
            )
            trainer.validate(model=model, dataloaders=[val_loader])


# main(DATASET='shakespeare', DEVICE=1, NUM_PREFIX=4, base='byte', do_e2e=True, EPOCHS=1, debug=True)

# for base in ['byte','char','sub','word']:
#     for e2e in [True, False]:
#         for dataset in ['indic-hi']:
#             if base == 'word' and e2e:
#                 continue
#             print(f"{'-'*20} {dataset=} {base=} {e2e=} {'-'*20}")
#             main(DATASET=dataset, DEVICE=1, NUM_PREFIX=4, base=base, do_e2e=e2e, EPOCHS=1, debug=True, block_size=128+64, batch_size=2, )

# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/wllwtqb7/checkpoints/epoch=38-step=20436.ckpt", DEVICE=2, DATASET="shakespeare") # 4sub
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/x03sqg3v/checkpoints/epoch=42-step=22139.ckpt", DEVICE=2, DATASET="shakespeare") # 4char
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/rs1h5q13/checkpoints/epoch=42-step=22139.ckpt", DEVICE=2, DATASET="shakespeare") # 4byte
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/w5d7jjq5/checkpoints/epoch=49-step=26200.ckpt", DEVICE=2, DATASET="shakespeare") # word - eval exists
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/8t2bx02p/checkpoints/epoch=45-step=23711.ckptt", DEVICE=2, DATASET="shakespeare") # 1sub
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/prwgihkd/checkpoints/epoch=49-step=26200.ckpt", DEVICE=2, DATASET="shakespeare") # 1char
# main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/2vrul1ay/checkpoints/epoch=49-step=26200.ckpt", DEVICE=2, DATASET="shakespeare") # 1byte


# _sub, _char, _byte: word 0
# main(LOAD_CKPT="/scratch1/sghaneka/etok/checkpoints/etok/23fb899d588d450e8bc8a7fc4e83496a/checkpoints/epoch=49-step=44898.ckpt", DEVICE=3, DATASET="shakespeare") # _sub
# main(LOAD_CKPT="/scratch1/sghaneka/etok/checkpoints/etok/deed7582996b4c11814314c71a363498/checkpoints/epoch=49-step=44898.ckpt", DEVICE=3, DATASET="shakespeare") # _char
# main(LOAD_CKPT="/scratch1/sghaneka/etok/checkpoints/etok/18970dc331f24b999edbd5d17a33f555/checkpoints/epoch=49-step=44898.ckpt", DEVICE=3, DATASET="shakespeare") # _byte

# old
# main(LOAD_CKPT="/scratch1/sghaneka/etok/checkpoints/etok/8c25cc8dcdc04bf49aad54a77f8046f4/checkpoints/epoch=49-step=9800.ckpt", DEVICE=3, DATASET="shakespeare") # sub
# main(LOAD_CKPT="/scratch1/sghaneka/etok/checkpoints/etok/cbf7310be4a24176b4d8cd28879171f4/checkpoints/epoch=49-step=9800.ckpt", DEVICE=2, DATASET="shakespeare") # 4byte


def boolean_string(s: str) -> bool:
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-base", type=str)  # byte, char, sub, word
    parser.add_argument("--ckpt", type=str, default="")  #
    parser.add_argument("-dataset", type=str)  # shakespeare, mc4, trial, oscar
    parser.add_argument("--num_prefix", type=int, default=4)
    parser.add_argument("--lang", type=str, default="en")  # en, gu, hi, zh
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fixed_context_char_size", type=int, default=150)
    parser.add_argument("--learning_rate", type=float, default=50)
    parser.add_argument("--block_size", type=int, default=128 + 64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--use_logger", type=str, default="true")
    parser.add_argument("--report_numeracy", type=boolean_string, default="false")
    parser.add_argument("--e2e", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    if args.ckpt == "":
        main(
            DATASET=args.dataset,
            DEVICE=args.device,
            NUM_PREFIX=args.num_prefix,
            base=args.base,
            do_e2e=args.e2e,
            EPOCHS=args.num_epochs,
            block_size=args.block_size,
            batch_size=args.batch_size,
            debug=False,
            LANG=args.lang,
            report_numeracy=args.report_numeracy,
            learning_rate=args.learning_rate,
            fixed_context_char_size=args.fixed_context_char_size,
        )
    else:
        print(args.use_logger)
        main(
            LOAD_CKPT=args.ckpt,
            DATASET=args.dataset,
            DEVICE=args.device,
            LANG=args.lang,
            USE_LOGGER=args.use_logger,
            fixed_context_char_size=args.fixed_context_char_size,
        )
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/ft7nhoax/checkpoints/epoch=76-step=40217.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/hwwmln3w/checkpoints/epoch=86-step=45195.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4char
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/ora0kiht/checkpoints/epoch=85-step=44671.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4byte

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/qk5wxmlm/checkpoints/epoch=99-step=52400.ckpt", DEVICE=args.device, DATASET="shakespeare") # word

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/14928kri/checkpoints/epoch=14-step=34800.ckpt", DEVICE=args.device, DATASET="shakespeare") # noe2e_sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/mumj32er/checkpoints/epoch=20-step=48600.ckpt", DEVICE=args.device, DATASET="shakespeare") # noe2e_char
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/ntoq12p7/checkpoints/epoch=11-step=27000.ckpt", DEVICE=args.device, DATASET="shakespeare") # noe2e_byte

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/j628fm00/checkpoints/epoch=87-step=46112.ckpt", DEVICE=args.device, DATASET="shakespeare") # 1sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/lo2qujr5/checkpoints/epoch=87-step=45981.ckpt", DEVICE=args.device, DATASET="shakespeare") # 1char
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/ssfcmzo9/checkpoints/epoch=81-step=42575.ckpt", DEVICE=args.device, DATASET="shakespeare") # 1byte

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/hs1qv3ow/checkpoints/epoch=49-step=13098.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4sub-batch_size=4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/8mi6q7yr/checkpoints/epoch=49-step=13098.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4char-batch_size=4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/uvhigs8o/checkpoints/epoch=49-step=13098.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4byte-batch_size=4

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/v7w13v7p/checkpoints/epoch=39-step=20567.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4sub-lr=7e-4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/q41lk0pd/checkpoints/epoch=43-step=22794.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4char-lr=7e-4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/wpr39fxi/checkpoints/epoch=41-step=22008.ckpt", DEVICE=args.device, DATASET="shakespeare") # 4byte-lr=7e-4

        #############-------------------------------------------------#############

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/hzad2ty5/checkpoints/epoch=9-step=29664.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # word
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/qpg09iqd/checkpoints/epoch=0-step=17180.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # noe2e_sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/q49q14fj/checkpoints/epoch=0-step=12885.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # noe2e_byte VALIDATION ON THIS WORKING. GENERATING

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/o08mnki9/checkpoints/epoch=0-step=3204.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/65e0u8ch/checkpoints/epoch=1-step=6411.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4char VALIDATION ON THIS WORKING. GENERATING TAKING AROUNG 40ish mintues.
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/1h7e0l52/checkpoints/epoch=0-step=1602.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4byte

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/2pxzl34h/checkpoints/epoch=0-step=1604.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4sub-batch_size=4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/39qjvl5e/checkpoints/epoch=2-step=3609.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4char-batch_size=4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/0cak0zq7/checkpoints/epoch=0-step=802.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4byte-batch_size=4

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/e33ze7zy/checkpoints/epoch=0-step=3204.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 1sub
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/n7tninhk/checkpoints/epoch=2-step=7215.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 1char
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/9ouphfnt/checkpoints/epoch=0-step=2403.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 1byte

        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/tugmwyyx/checkpoints/epoch=0-step=1604.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4sub-lr=7e-4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/k1una73m/checkpoints/epoch=2-step=3609.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4char-lr=7e-4
        # main(LOAD_CKPT="/scratch1/sghaneka/etok/etok/a5nrja96/checkpoints/epoch=0-step=802.ckpt", DEVICE=args.device, DATASET="custom", LANG="fr") # 4byte-lr=7e-4

        # main(LOAD_CKPT=f"/scratch1/sghaneka/etok/checkpoints/etok/{args.ckpt}/checkpoints/epoch=49-step=9800.ckpt", DEVICE=args.device, DATASET="shakespeare") # word
# nohup python unitrain.py -dataset shakespeare -base sub --no-e2e --device 2 &
