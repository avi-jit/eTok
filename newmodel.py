import math
import logging

import torch

import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np

from mingpt.model import Block

logger = logging.getLogger(__name__)


class myGPT(pl.LightningModule):
    """  end-to-end tokenized full GPT language model, with a context size of block_size """
    def __init__(self,
                 #e2e_vocab_size=100,
                 num_prefix=0, # 0 for no compression
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_head=4,
                 n_layer=12,
                 n_e2e_head=4,
                 n_e2e_layer=2,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 vocab=None,
                 base='char', # 'char' 'sub' 'word' 'byte'
                 canvas_size = 12,
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature

        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams
        self.vocab_length = len(vocab)
        
        empt = {v for k, v in vocab.items() if k == ' '} # ' ' is padding
        self.empt = empt

        if base == 'sub':
            news = {v for k,v in vocab.items() if k.startswith('Ġ')} # (?) shouldn't they all be first token for each c
            self.news = torch.tensor(list(news), dtype=torch.int64).unsqueeze(0)
        elif base in ['char','word']:
            self.rev = {k:v for v,k in vocab.items()}

        # end-to-end tokens
        if num_prefix > 0: # only prefix = 1 is considered
            assert base != 'word'
            self.wordenc = nn.Sequential(*[Block(self.config) for _ in range(n_e2e_layer)])
            canvas_size = canvas_size
            self.word_pe = nn.Parameter(torch.zeros(1, block_size*num_prefix, n_embd))
        else:
            canvas_size = block_size
        
        # input embedding stem
        self.in_emb = nn.Embedding(len(vocab) + num_prefix, n_embd)
        self.in_pe = nn.Parameter(torch.zeros(1, canvas_size + num_prefix, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.config.block_size = block_size * max(num_prefix,1)
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        self.config_block_size = block_size

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        if num_prefix > 0:
            self.config.block_size = canvas_size + num_prefix 
            self.decoder_blocks = nn.Sequential(*[Block(self.config) for _ in range(n_e2e_layer)])
            self.config.block_size = block_size 
            self.head = nn.Linear(n_embd, len(vocab), bias=False)
        else:
            self.head = nn.Linear(n_embd, len(vocab), bias=False)
            self.acc= {k:torchmetrics.Accuracy(top_k=k,mdmc_average='global') for k in [1,5,25]}

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _add_cls(self, config, seq) :
        cls_idx = []
        for i in range(len(seq)) :
            if(seq[i] in self.news) :
                seq = seq[:i]+[range(self.vocab_length, self.vocab_length+config.num_prefix)-1]+seq[i:] # prefix index is set to vocab_len -> vocab_len+num_pre-1
        for i in range(len(seq)):
            if(seq[i] in self.news):
                cls_idx.append(i-config.num_prefix)
        return seq, cls_idx
    
    def _make_mask(self, m_len, cls_idx): #m_len: matrix length
        mask = torch.zeros((m_len, m_len)) # (?) can we write like this?
        for j in range(len(cls_idx)):
            now = cls_idx[j]
            nxt = m_len
            if(not (j == len(cls_idx)-1)) : # if not last triangle
                nxt = cls_idx[j+1]
            for tri_i in range(now, nxt):
                for tri_j in range(now, now+1+(tri_i-now)):
                    mask[tri_i][tri_j] = 1
        return mask

    def _extract_cls(self, out, cls_idx):
        num_cls = len(cls_idx)*self.config.num_prefix
        cls_out = out[cls_idx[0]:cls_idx[0]+self.config.num_prefix] # [t, D]
        for i in range(1, cls_idx):
            cls_out = torch.cat((cls_out, out[cls_idx[i]:cls_idx[i]+self.config.num_prefix]), dim=0)
        
        return cls_out, num_cls

    def _update_cls(self, now_embd, net_list,cls_idxs):
        for i in range(len(cls_idxs)):
            init_cls_idx = (i-1)*self.config.num_prefix
            init_embd_idx = cls_idxs[i]
            for shift in range(self.config.num_prefix):
                now_embd[init_embd_idx+shift] = net_list[init_cls_idx+shift]

        return now_embd

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.config.block_size
                
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

# since tensor has to have same dimensionality -> sequences are padded
    def forward(self, idx, mask=None, eval=False):
        p = self.config.num_prefix
        assert idx.size()[1] <= self.config.block_size, "Cannot forward, model block size is exhausted."

        cls_idxs = [] # the index for cls tokens (using the first cls before each head token)
        if p > 0:
            b, t = idx.size()
            if(self.base == 'sub'):
                new_idxs = []
                token_embeddings = []
                for i in range(b):
                    new_idx, cls_idx = self._add_cls(idx[i,:]) # the dimensionality is different for each t now
                    new_idxs.append(new_idx)
                    cls_idxs.append(cls_idx)
                    token_embeddings.append(self.in_emb[new_idx]) #[B, t_new, D]
                
                for i in range(b):
                    in_pe = torch.squeeze(self.in_pe[:, len(new_idxs[i]), :]) # (t, D)
                    token_embeddings[i] = self.drop(token_embeddings[i] + in_pe) # (?) does the dimensionality work?
            else:   # B, t, D
                token_embeddings = self.emb(idx)
                in_pe = self.in_pe[:,:t].unsqueeze(-1)
                token_embeddings = self.drop(token_embeddings + in_pe) # [b, t_new, -1]
            
            # process each sequence individually (different dimensionality)
            
            in_emb_list = []
            for i in range(b):
                # make mask for each sequence
                now_mask = self._make_mask(len(new_idx[i]), cls_idxs[i])
                input = token_embeddings[i].unsqueeze(0) #[1, t, D]
                out = self.wordenc(input, now_mask) #[1, t, D]
                out = torch.squeeze(out) #[1, t, D] -> [t, D]
                out, num_cls = self._extract_cls(out) # only maintain cls
                word_pe = torch.squeeze(self.word_pe[:, :num_cls, :]) #[1, t, D] -> [t, D]
                now_in_emb = self.drop(out + word_pe) 
                in_emb_list.append(now_in_emb) #for each element: [t, D] 

        else:
            b, t = idx.size()
            in_emb = self.in_emb(idx)
        
        if(p > 0):
            net_list = []
            for i in range(len(in_emb_list)):
                seq_len = len(in_emb_list[i])
                now_net = self.blocks(in_emb_list.unsqueeze(0), self.mask[:seq_len, :seq_len]) 
                now_net = self.ln_f(now_net)
                now_net = torch.squeeze(now_net)
                net_list.append(now_net) # [B, t, D]
        else:
            net = self.blocks(in_emb)
            net = self.ln_f(net)
        
        if p > 0:
            if eval:
                return net_list # [b, t_new, D]
            
            canvas = token_embeddings # [b, t, D]

            logits = []
            # use new net replace old tokens
            for i in range(b):
                canvas[i] = self._update_cls(canvas[i], net_list[i], cls_idxs[i])
                in_pe = torch.squeeze(self.in_pe[:, len(canvas[i]), :])
                now_token_out = torch.unsqueeze(self.drop(canvas[i] + in_pe),0) # [1, t, D]
                seq_len = len(canvas[i])
                out = self.decoder_blocks(now_token_out, self.mask[:seq_len, :seq_len]) # [1, t, D]
                now_logit = self.head(out)
                logits.append(now_logit)
        else:
            logits = self.head(net.reshape(b,t,-1))

        return logits

    # def training_step(self, batch, batch_idx, eval=False):
    #     if self.config_num_prefix == 0:
    #         x, y = batch
    #         logits = self(x)
    #     else:
    #         x, y, x_mask, y_mask = batch
    #         logits = self(x, x_mask)  # change forward to adapt to input mask
        
    #     loss = None
    #     if y is not None:
    #         if self.config.num_prefix > 0:
    #             loss = F.cross_entropy() # change cross_entropy to adapt to different dimensionality
                    
    
