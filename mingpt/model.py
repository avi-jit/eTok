"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch

import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head

    def forward(self, x, mask, layer_past=None):
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, T, T]
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class eGPT(pl.LightningModule):
    """  end-to-end tokenized full GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size,
                 out_vocab_size,
                 num_heads=4,
                 e2e_vocab_size=100,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 ctoi=None,
                 itoc=None,
                 wtoi=None,
                 itow=None,
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # end-to-end tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=8, batch_first=True)
        self.wordenc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.e2e_emb = torch.nn.Parameter(torch.Tensor(e2e_vocab_size, n_embd)) #nn.Embedding(e2e_vocab_size, n_embd)
        #torch.nn.init.xavier_uniform_(self.e2e_emb, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.normal_(self.e2e_emb, mean=0.0, std=1.0)
        self.multihead_attn = nn.MultiheadAttention(n_embd, num_heads, batch_first=True)

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.char_pe = nn.Parameter(torch.zeros(1, 50, n_embd)) # 21 will do
        self.word_pe = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.out_vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        self.acc= {k:torchmetrics.Accuracy(top_k=k,mdmc_average='global') for k in [1,5,25]}
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

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

    def forward(self, idx, mask, eval=False):
        b, t, c = idx.size(); # mask is b,t. needs to be (4096 = 8 heads * 512 = 4 b * 128 t, 21, 21) or (21, 21). 
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        char_pe = self.char_pe[:, :c, :].unsqueeze(1) # each position maps to a (learnable) vector
        chars = self.drop(token_embeddings + char_pe) # B,t,D. so far 1 word 1 embedding.

        # get query
        mask = mask.reshape(-1) # b*t
        mask = torch.arange(c, device=mask.device).expand(len(mask), c) > mask.unsqueeze(1) # b*t,c
        mask0 = torch.einsum('bi,bj->bij', mask, mask)
        mask1 = torch.einsum('bi,bj->bij', 1-mask*1, 1-mask*1)<1 # small square of False, rest True which are later mapped to -inf

        #out = self.wordenc(chars.reshape(b*t,c,-1), mask=mask.repeat(self.config.n_head,1,1)) # mask is b*t*h,c,c. out is b*t,c,D same as in... here 1 word many embeddings. dropout in encoder? why in eval mode?
        out = self.wordenc(chars.reshape(b*t,c,-1), mask=torch.repeat_interleave(mask0,self.config.n_head,dim=0))
        query = out[:,0,:] # first denotes CLS per word
        attn_output, attn_output_weights = self.multihead_attn(query, self.e2e_emb.to(query.device), self.e2e_emb.to(query.device), average_attn_weights=False) # ow=b*t,Ve
        
        # todo: position embeddings word level!
        word_pe = self.word_pe[:, :t, :] # each position maps to a (learnable) vector
        words = self.drop(attn_output.reshape(b,t,-1) + word_pe)

        x = self.blocks(words) # b,t,d
        x = self.ln_f(x)
        logits = self.head(x)
        if eval:
            return (logits, attn_output_weights.reshape(self.config.num_heads, b, t, self.config.e2e_vocab_size),
                        query.reshape(b, t, -1))
        return logits

    def training_step(self, batch, batch_idx):
        idxs, targets, masks = batch
        # same action as inference
        logits = self(idxs, masks) # b,t,Vw

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        idxs, targets, masks = batch
        # same action as inference
        logits = self(idxs, masks) # b,t,wvocab

        # if we are given some desired targets also calculate the loss
        #loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            #acc = torch.argmax(logits,dim=-1)==targets # masks are for chars, not words
            #self.log("val_acc", acc.sum()/acc.numel(), prog_bar=True)
            for k,v in self.acc.items():
                self.log(f"val_acc@{k}", v.to(targets.device)(logits.transpose(1,2), targets), prog_bar=True)
            self.log("val_loss", loss, prog_bar=True)

class eGPT_pre(pl.LightningModule):
    """  end-to-end tokenized full GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size, # chars
                 out_vocab_size=None, # leave none for NAT; give value for word decoder
                 num_heads=4,
                 #e2e_vocab_size=100,
                 num_prefix=1,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 nat_layers=2,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 ctoi=None,
                 itoc=None,
                 wtoi=None,
                 itow=None,
                 bigram=None,
                 decoder='word', # 'word' 'nat' 'ar'
                 base='none',
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # end-to-end tokens
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
        self.wordenc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        #self.e2e_emb = torch.nn.Parameter(torch.Tensor(e2e_vocab_size, n_embd)) #nn.Embedding(e2e_vocab_size, n_embd)
        #torch.nn.init.xavier_uniform_(self.e2e_emb, gain=torch.nn.init.calculate_gain("linear"))
        #torch.nn.init.normal_(self.e2e_emb, mean=0.0, std=1.0)
        #self.multihead_attn = nn.MultiheadAttention(n_embd, num_heads, batch_first=True)

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size + num_prefix, n_embd)
        self.char_pe = nn.Parameter(torch.zeros(1, 12, n_embd)) # 21 will do
        #self.word_pe = nn.Parameter(torch.zeros(1, block_size, n_embd*num_prefix))
        self.word_pe = nn.Parameter(torch.zeros(1, block_size*num_prefix, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        
        # transformer
        #self.config.n_embd = n_embd * num_prefix
        self.config.block_size = block_size * num_prefix # for blocks
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        self.config.block_size = block_size # back to normal; to save this in ckpt

        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        if decoder == 'word':
            #if out_vocab_size: # word decoder
            #self.head = nn.Linear(self.config.n_embd * self.config.num_prefix, self.config.out_vocab_size, bias=False)
            self.head = nn.Sequential(nn.Linear(self.config.n_embd * self.config.num_prefix, 218,bias=False), nn.Linear(218,self.config.out_vocab_size, bias=False))
            self.acc= {k:torchmetrics.Accuracy(top_k=k,mdmc_average='global') for k in [1,5,25]}
        elif decoder == 'nat': # nat decoder
            #self.config.n_embd = n_embd # just needed for Block call and ln_f
            decoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, batch_first=True)
            self.worddec = nn.TransformerEncoder(decoder_layer, num_layers=nat_layers)
            self.head = nn.Linear(self.config.n_embd, vocab_size, bias=False)
            if bigram:
                self.bigram = torch.ones(self.config.vocab_size+1, self.config.vocab_size)
                for (i,j),f in bigram.items():
                    self.bigram[ctoi.get(i,281),ctoi.get(j)] += f
                self.bigram = torch.log(F.normalize(self.bigram, p=1, dim=0))
        elif decoder == 'ar':
            self.config.block_size = 10 + num_prefix # for blocks
            self.decoder_blocks = nn.Sequential(*[Block(self.config) for _ in range(nat_layers)])
            self.config.block_size = block_size # back to normal; to save this in ckpt
            self.head = nn.Linear(self.config.n_embd, vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.decoder = decoder
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

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

    def forward(self, idx, mask, eval=False):
        p = self.config.num_prefix
        b, t, c0 = idx.size(); # mask is b,t. needs to be (4096 = 8 heads * 512 = 4 b * 128 t, 21, 21) or (21, 21). 
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        # prefix the idx
        mask += self.config.num_prefix
        prefix = torch.tensor(range(c0,c0+p,), device=idx.device)
        x = torch.cat((prefix.unsqueeze(0).unsqueeze(0).repeat(b,t,1),idx),-1)
        b, t, c = x.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(x) # each index maps to a (learnable) vector
        char_pe = self.char_pe[:, :c, :].unsqueeze(1) # each position maps to a (learnable) vector
        chars = self.drop(token_embeddings + char_pe) # B,t,D. so far 1 word 1 embedding.

        # get query
        mask = mask.reshape(-1) # b*t
        mask = torch.arange(c, device=mask.device).expand(len(mask), c) > mask.unsqueeze(1) # b*t,c
        mask0 = torch.einsum('bi,bj->bij', mask, mask)
        #mask1 = torch.einsum('bi,bj->bij', 1-mask*1, 1-mask*1)<1 # small square of False, rest True which are later mapped to -inf

        #out = self.wordenc(chars.reshape(b*t,c,-1), mask=mask.repeat(self.config.n_head,1,1)) # mask is b*t*h,c,c. out is b*t,c,D same as in... here 1 word many embeddings. dropout in encoder? why in eval mode?
        out = self.wordenc(chars.reshape(b*t,c,-1), mask=torch.repeat_interleave(mask0,self.config.n_head,dim=0))
        query = out[:,:p,:] # first few denote CLS per word. 2560,4,512 = b*t,p,d. earlier was b*t,d
        #attn_output, attn_output_weights = self.multihead_attn(query, self.e2e_emb.to(query.device), self.e2e_emb.to(query.device), average_attn_weights=False) #ow=b*t,Ve
        
        # todo: position embeddings word level!
        word_pe = self.word_pe[:, :t*p, :] # 1,128,512 = 1,t,d
        #words = self.drop(query.reshape(b,t,-1) + word_pe) # b,t,p*d i.e. all p prefixes concatenated into one
        words = self.drop(query.reshape(b,t*p,-1) + word_pe) # b,t*p,d i.e. all p prefixes are separate tokens. way costlier due to O(n2) attn

        net = self.blocks(words) # b,t*p,d - todo: try variant where each word gets 1 token p*d
        net = self.ln_f(net)

        if self.decoder == 'word':
            #if self.config.out_vocab_size:
            # decoding words
            logits = self.head(net.reshape(b,t,-1)) # b,t*p,d -> b,t,Vw
        elif self.decoder == 'nat': # nat decoder
            # decoding characters
            canvas = torch.zeros((b*t,c), dtype=torch.int, device=net.device) # 0 is blank.
            canvas = self.tok_emb(canvas) # b*t,c,d
            canvas[:,:p] = net.reshape(b*t,p,-1)
            char_pe = self.char_pe[:, :c, :] # 1,c,d
            canvas = self.drop(canvas + char_pe)
            painted = self.worddec(canvas) # b*t,c,d
            logits = self.head(painted)[:,p:] # b*t,c,Vc to b*t,c0,Vc
            #torch.cat(net.reshape(b*t,self.config.num_prefix,-1),)
            #logits = self.head(x)
            logits = logits.reshape(b,t,c0,-1)
            #L[:,:,1:] += torch.einsum('blnc,cd->blnd', L[:,:,:-1], torch.log(F.normalize(self.bigram[:-1,:].to(logits.device), p=1, dim=-1)) )
            # L.argmax(-1)[0,5]
            #logits[:,:,0] += torch.log(self.bigram[-1].to(logits.device))
            #logits[:,:,1:] *= self.bigram[-1].to(logits.device)
            '''L = logits.clone()
            for i in range(1,10):
                for j in range(0,281):
                    L[:,:,i,j] = torch.logsumexp(L[:,:,i-1]+self.bigram[:-1,j].to(logits.device), dim=-1)
            L[:,:,0] = self.bigram[-1].to(logits.device)
            logits += L*20.0'''
            # [self.config.itoc[_] for _ in (logits*0.5 + L).argmax(-1)[0,5].cpu().tolist()]
            #for i in range(1,10):
            #    logits[:,:,i,:] *= torch.einsum('blc,cd->bld', logits[:,:,i-1], self.bigram[:-1,:].to(logits.device))
            #logits[:,:,1:] *= torch.einsum('blnc,cd->blnd', logits[:,:,:-1], self.bigram[:-1,:].to(logits.device))
            #logits[:,:,1:] *= torch.einsum('blnc,cd->blnd', logits[:,:,:-1], F.normalize(self.bigram[:-1,:].to(logits.device), p=1, dim=-1) ) # weighted mean
        elif self.decoder == 'ar':
            # net is (b,t*p,d). use these to replace first p of chars: (b,t,c=p+c0,d)
            canvas = token_embeddings[:,1:,:,:] # b,t-1,c,d
            canvas[:,:,:p,:] = net.reshape(b,t,p,-1)[:,:-1,:,:]
            chars_out = self.drop(canvas + char_pe)
            out = self.decoder_blocks(chars_out.reshape(b*(t-1),c,-1))
            logits = self.head(out.reshape(b,t-1,c,-1)[:,:,p-1:-1,:]) # b,t-1,c0,Vc
        if eval:
            return (logits, query.reshape(b, t, p, -1)) # b,t,c,Vc and b,t,p,d
        return logits

    def training_step(self, batch, batch_idx):
        if self.config.out_vocab_size:
            x, y, x_mask = batch
        else:
            x, y, x_mask, y_mask = batch
        #idxs, targets, masks = batch
        # same action as inference
        logits = self(x, x_mask) # b,t,wvocab

        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if self.decoder == 'nat':
                loss = F.cross_entropy(logits.transpose(1,2).transpose(1,3), y)
                '''loss = F.cross_entropy(logits.transpose(1,2).transpose(1,3), y, reduction='none') # b,t,c
                loss = torch.prod(2+loss, dim=-1)**(1/10) # b,t
                loss = loss.mean()
                loss = loss/4 + F.cross_entropy(logits.transpose(1,2).transpose(1,3), y) # orig+GM/4'''
            elif self.decoder == 'ar':
                loss = F.cross_entropy(logits.transpose(1,2).transpose(1,3), y[:,:-1,:]) # logits (b,t-1,c0,Vc). y (b,t,c0)
            elif self.decoder == 'word':
                loss = F.cross_entropy(logits.transpose(1,2), y)    
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.decoder == 'word':
            x, y, x_mask = batch
            b, t, c = x.size()
        elif self.decoder in ['nat', 'ar']:
            x, y, x_mask, y_mask = batch       
            if self.decoder == 'ar' and y is not None:
                y = y[:,:-1,:] # logits (b,t-1,c0,Vc). y (b,t,c0)
                y_mask = y_mask[:,:-1] 
            b, t, c = y.size()
        #idxs, targets, masks = batch
        # same action as inference
        logits = self(x, x_mask) # b,t,wvocab

        # if we are given some desired targets also calculate the loss
        #loss = None
        if y is not None:
            if self.decoder in ['nat','ar']:
                loss = F.cross_entropy(logits.transpose(1,2).transpose(1,3), y)
            else:
                loss = F.cross_entropy(logits.transpose(1,2), y)
            self.log("val_loss", loss, prog_bar=True)
            if self.decoder == 'word':
                for k,v in self.acc.items():
                    self.log(f"val_acc@{k}", v.to(y.device)(logits.transpose(1,2), y), prog_bar=True)
            elif self.decoder in ['nat', 'ar']:
                acc_c0 = torch.argmax(logits,dim=-1)==y # b,t,c
                acc_w0 = torch.all(acc_c0, dim=2)

                # ignore 0s
                mask = y_mask.reshape(-1)
                #mask = y_mask.view(-1)
                mask = torch.arange(c, device=mask.device).expand(len(mask), c) < mask.unsqueeze(1) # True for non0
                mask = mask.view(b,t,c)
                acc_c = acc_c0 * mask # b,t,c
                return acc_c0, acc_c, acc_w0, mask
            
    def validation_epoch_end(self, outputs) -> None:
        if outputs==[]:
            return
        acc_c0 = torch.cat([_[0] for _ in outputs]); acc_c = torch.cat([_[1] for _ in outputs]); acc_w0 = torch.cat([_[2] for _ in outputs]); 
        mask = torch.cat([_[3] for _ in outputs]);
        self.log("val_acc_char+0", acc_c0.sum()/acc_c0.numel(), prog_bar=True) # overestimate acc because 0 is easy
        self.log("val_acc_char", acc_c.sum()/mask.sum(), prog_bar=True)
        self.log("val_acc_word+0", acc_w0.sum()/acc_w0.numel(), prog_bar=True) # fair acc because 0 is needed


class ByT5(pl.LightningModule):
    """  char level GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size, # chars or subs
                 num_heads=4,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 nat_layers=2,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 ctoi=None,
                 itoc=None,
                 base='char',
                 **kwargs,
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams
        self.config.num_prefix=''

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.char_pe = nn.Parameter(torch.zeros(1, block_size, n_embd)) # 21 will do
        self.drop = nn.Dropout(embd_pdrop)
        
        # transformer
        self.config.block_size = block_size # for blocks
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        self.config.block_size = block_size # back to normal; to save this in ckpt

        # decoder head
        self.head = nn.Linear(self.config.n_embd, vocab_size, bias=False)
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.acc= {k:torchmetrics.Accuracy(top_k=k,mdmc_average='global') for k in [1,5,25]}
        self.block_size = self.config.block_size
        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def get_block_size(self):
        return self.block_size
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
        p = self.config.num_prefix
        b, t = x.size();
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        # forward the GPT model
        token_embeddings = self.tok_emb(x) # each index maps to a (learnable) vector
        char_pe = self.char_pe[:, :t, :] # each position maps to a (learnable) vector
        chars = self.drop(token_embeddings + char_pe) # B,t,D. so far 1 word 1 embedding.

        net = self.blocks(chars) # b,t,d
        net = self.ln_f(net)

        logits = self.head(net) # b,t,d -> b,t,Vc
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # b,t,Vc

        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # b,t,Vc

        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            self.log('val_loss', loss)
            acc_c0 = torch.argmax(logits,dim=-1)==y # b,t,c
            #for k,v in self.acc.items():
            #    self.log(f"val_acc_char@{k}", v.to(y.device)(logits.transpose(1,2), y), prog_bar=True)
            #acc_w0 = torch.all(acc_c0, dim=2)
            return acc_c0#, acc_c, acc_w0
            
    def validation_epoch_end(self, outputs) -> None:
        if outputs==[]:
            return
        acc_c0 = torch.cat([_[0] for _ in outputs])#; acc_c = torch.cat([_[1] for _ in outputs]); acc_w0 = torch.cat([_[2] for _ in outputs]); 
        #mask = torch.cat([_[3] for _ in outputs]);
        self.log(f"val_acc_{self.config.base}", acc_c0.sum()/acc_c0.numel(), prog_bar=True) # overestimate acc because 0 is easy
        #self.log("val_acc_char", acc_c.sum()/mask.sum(), prog_bar=True)
        #self.log("val_acc_word+0", acc_w0.sum()/acc_w0.numel(), prog_bar=True) # fair acc because 0 is needed


class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self,
                 vocab_size,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

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

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        # same action as inference
        logits = self(idx)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        #result = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        
        #result.log('train_loss', loss)
        self.log('train_loss', loss)
        return loss

