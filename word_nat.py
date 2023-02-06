import torch
import torch.nn as nn
from torch.nn import functional as F
# make deterministic
import pytorch_lightning as pl
pl.seed_everything(42)
import regex as re
from tqdm import tqdm 
import wandb
import datasets
from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback

import collections
from torch.utils.data import DataLoader

batch_size = 2**13 # 8 # 20 # 4096 = 2**12
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
DATASET='wiki' 
#DATASET='shakespeare'
DEVICE=3

if DATASET == 'shakespeare': # one line of poem is roughly 50 characters
    text = open('/nas/home/thawani/etok/tinyshake.txt', 'r').read() # don't worry we won't run out of file handles
elif DATASET == 'wiki':
    text = ' '.join(datasets.load_dataset("wikitext", "wikitext-2-v1", split="train", )['text'])
else:
    raise NotImplementedError

class eDataset_wordnat(torch.utils.data.Dataset):
    def __init__(self, data, word_vocab_size=None):
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r' +',r' ',text).strip()
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        self.data = text.split(' ')
        self.data = [_ if 1<=len(_)<=9 else "@" for _ in self.data] # removes 4.6% tokens
        words = list(set(self.data))
        self.maxlen = max(len(_) for _ in words) # max number of chars in a word
        print(f"{self.maxlen=}")

        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data), len(words)))
        self.ctoi = { ch:i for i,ch in enumerate(chars) }
        self.itoc = { i:ch for i,ch in enumerate(chars) }
        
        fwords = collections.Counter(self.data).most_common(word_vocab_size-1)  
        print(f'Top {word_vocab_size-1} words cover {100*sum([_[1] for _ in fwords])/len(self.data):.2f}% of all words')
        self.wtoi = collections.defaultdict(lambda w:0)
        self.itow = collections.defaultdict(lambda i:'UNK') 
        words = [_[0] for _ in fwords]
        for i,w in enumerate(words):
            self.wtoi[w] = i+1
            self.itow[i+1] = w
        self.wvocab_size = word_vocab_size   
        self.cvocab_size = len(chars)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        word = self.data[idx]
        x = self.wtoi.get(word,0)
        y = [self.ctoi[_] for _ in word]+[0]*(9-len(word))
        return torch.tensor(x), torch.tensor(y)

full_dataset = eDataset_wordnat(text, word_vocab_size=1_000)
# use 20% of training data for validation
print(full_dataset[0])
train_set_size = int(len(full_dataset) * 0.8)
valid_set_size = len(full_dataset) - train_set_size

# split the train set into two
#seed = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(full_dataset, [train_set_size, valid_set_size])

train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=16)
val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=16)

# val_set how many 0s? 60% of 9*400k

class eGPT_wordnat(pl.LightningModule):
    """  decodes words to characters """
    def __init__(self,
                 num_prefix=1,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 embd_pdrop=0.1,
                 nat_layer=6,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 ctoi=None,
                 itoc=None,
                 wtoi=None,
                 itow=None,
                 ):
        super().__init__()
        char_vocab_size = len(ctoi) # chars
        word_vocab_size = len(wtoi)+1 # words
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()
        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # input embedding stem
        self.word_emb = nn.Embedding(word_vocab_size, n_embd * num_prefix) 
        self.char_emb = nn.Embedding(1, n_embd) 
        self.char_pe = nn.Parameter(torch.zeros(1, 9+num_prefix, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        decoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
        self.worddec = nn.TransformerEncoder(decoder_layer, num_layers=nat_layer)
        self.head = nn.Linear(n_embd, char_vocab_size, bias=False)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
        return optimizer

    def forward(self, x, eval=False):
        B, = x.shape # one word per item
        D = self.config.n_embd
        K = self.config.num_prefix
        C = 9
        net = self.word_emb(x) # B,D*K
        net = net.reshape(B,K,-1)

        canvas = torch.zeros((B,K+C), dtype=torch.int, device=net.device) # 0 is blank.
        canvas = self.char_emb(canvas) # B,K+C,D
        canvas[:,:K] = net
        char_pe = self.char_pe[:, :K+C, :] # 1,K+C,D
        canvas = self.drop(canvas + char_pe)
        painted = self.worddec(canvas) # B,K+C,D
        logits = self.head(painted[:,K:]) # B,C,D
        return logits

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        B,C,V = logits.shape
        loss = None # if we are given some desired targets also calculate the loss
        if y is not None:
            loss = F.cross_entropy(logits.transpose(1,2), y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        B,C,V = logits.shape
        loss = None # if we are given some desired targets also calculate the loss
        preds = torch.argmax(logits, dim=-1) # B,C
        pred0s = preds==0
        if y is not None:
            #loss = F.cross_entropy(logits.view(B,V,C), y)
            loss = F.cross_entropy(logits.transpose(1,2), y)
            acc_c = (preds==y)
        self.log('val_loss', loss, prog_bar=True)
        return pred0s, acc_c
        
    def validation_epoch_end(self, validation_step_outputs):
        pred0s = torch.cat([_[0] for _ in validation_step_outputs])
        acc_cs = torch.cat([_[1] for _ in validation_step_outputs])
        self.log('val_0s', pred0s.float().mean(), prog_bar=True) # % of 0s in C * total words in val_set 400k
        acc_w = torch.all(acc_cs, dim=-1)
        self.log('val_acc_c', acc_cs.float().mean(), prog_bar=True)
        self.log('val_acc_w', acc_w.float().mean(), prog_bar=True)
        # TODO: log acc per unique word?
        

model = eGPT_wordnat(
            num_prefix=1,
            nat_layer=2,
            n_head=8, 
            n_embd=512, 
            learning_rate=1e-4,
            itow=full_dataset.itow,
            wtoi=full_dataset.wtoi,
            itoc=full_dataset.itoc,
            ctoi=full_dataset.ctoi,
        )
model.word_emb.weight.requires_grad = False
if not model.word_emb.weight.requires_grad:
    frozen = 'frozen'
else:
    frozen = 'unfrozen'
# scheduler
lr_decay = LearningRateDecayCallback(learning_rate=1e-4, warmup_tokens=512*20,
                                        final_tokens=00*len(train_set)*1)
wandb_logger = pl.loggers.WandbLogger(project="etok")
wandb.run.name = f"wordnat {frozen} {DATASET} {'-'.join(wandb.run.name.split('-')[:2])}"

trainer = Trainer(
        profiler="simple",
        #accelerator="cpu",
        accelerator="gpu", devices=[DEVICE], 
        #precision=16, 
        max_epochs=10,
        gradient_clip_val=1.0, 
        callbacks=[lr_decay], 
        #progress_bar_refresh_rate=1, 
        #row_log_interval=1,
        log_every_n_steps=15,
        logger=wandb_logger,
        val_check_interval=0.25,
        num_sanity_val_steps=3,
        )
#trainer.fit(model, train_loader)
trainer.fit(model, train_loader, val_loader)
