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
                now_logit = torch.squeeze(self.head(out)) # [t, Vc]
                logits.append(now_logit) # # (!) understand what does this shifter mean!
        else:
            logits = self.head(net.reshape(b,t,-1))

        return logits

    def training_step(self, batch, batch_idx):
        if self.config.num_prefix == 0:
            x, y = batch
            logits = self(x)
        else:
            x, y, x_mask, y_mask = batch #(?) are masks here 2 dimensional boolean mask? are the masks at word/subword level?
            logits = self(x, x_mask) # [B, t, Vc] (!) add adaption to input masks
        
        loss = None
        if y is not None: # y [B, t, Vc]
            if self.config.num_prefix > 0:
                loss = []
                b, _, _=logits.size()
                for i in range(b):
                    loss.append(F.cross_entropy(logits[i], y[i]))
            else:
                loss = F.cross_entropy(logits.transpose(1,2), y)
        
        if not eval:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("gpu", torch.cuda.memory_allocated() / (1024 ** 3), on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:][-1][:]/temperature # get the last logits (the prediction)
            b, t, Vc = logits.size()
            probs = []
            if top_k is not None:
                v = []
                for i in range(b):
                    v, _ = torch.topk(logits[i], top_k) # v = [B, 1, top_k]
                    logits[i][logits[i] < v[:, [-1]]] = -float('Inf')
            
            for i in range(b):
                probs.append(F.softmax(logits[i], dim=-1))
            
            if do_sample:
                idx_next = torch.multinomidal(probs, num_samples=1) # (!) change to sequential
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1) 
        return idx
            
    def validation_step(self, batch, batch_idx):
        logging_batch_idx = -1 
        #self.log('val_loss', self.training_step(batch, batch_idx, eval=True), on_step=False, on_epoch=True, logger=True)
        context=90
        max_new_tokens=30
        if self.config.base == 'char':
            space_token = 0
        elif self.config.base == 'byte':
            space_token = 35
            #context=30
            max_new_tokens=60
        if self.config.num_prefix == 0 and self.config.base != 'word':
            x,y = batch
            b,l = x.shape
            
            inputs = torch.zeros((b,context), dtype=x.dtype, device=x.device)
            answers = torch.zeros((b,max_new_tokens), dtype=x.dtype, device=x.device)
            lens = torch.zeros(b, dtype=x.dtype, device=x.device)
            for i,row in enumerate(x):
                if self.config.base in ['char','byte']:
                    spaces = (row==space_token).nonzero()
                    last_word_beg, last_word_end = spaces[-2][0], spaces[-1][0]+1 # last full word
                    inputs[i] = row[last_word_beg - context+1:last_word_beg+1] # +1 to include space
                    answers[i,:last_word_end-last_word_beg-1] = row[last_word_beg+1:last_word_end]
                    lens[i] = last_word_end - last_word_beg -1 # 2. if gt is bank, banker should be wrong.
                elif self.config.base == 'sub':
                    spaces = (row.unsqueeze(1)==self.news.to(row.device)).any(dim=1).nonzero() # b size boolean, then nonzero
                    last_word_beg, last_word_end = spaces[-2][0], spaces[-1][0] # last full word
                    inputs[i] = row[last_word_beg - context:last_word_beg]
                    answers[i,:last_word_end-last_word_beg] = row[last_word_beg:last_word_end]
                    lens[i] = last_word_end - last_word_beg
            out = self.generate(inputs, max_new_tokens=max_new_tokens, temperature=1.0, do_sample=True, top_k=None)
            out = out[:,context:]
            mask = torch.arange(max_new_tokens, device=x.device).expand(len(lens), max_new_tokens) < lens.unsqueeze(1)
            corrects = (out==answers)*mask # if gt is bank, banker should be wrong.
            acc_unit = corrects.sum()/mask.sum() # sub/char
            acc_word = (corrects.float() + (1- mask.float())).bool().all(dim=-1).float().mean() # TODO: sub: penalize prefix match
            if batch_idx == logging_batch_idx:
                if self.config.base == 'char':
                    rows = [["".join([self.rev[_c] for _c in _context]).strip(), "".join([self.rev[_c] for _c in _pred]).strip(), "".join([self.rev[_c] for _c in _true]).strip()] for _context, _pred, _true in zip(inputs.cpu().tolist(), out.cpu().tolist(), answers.cpu().tolist()) ]
                elif self.config.base in ['sub','byte']:
                    rows = [[self.config.vocab.decode(_context), self.config.vocab.decode([_c for _c in _pred if _c!=0]), self.config.vocab.decode([_c for _c in _true if _c!=0]).strip() ] for _context, _pred, _true in zip(inputs.cpu().tolist(), out.cpu().tolist(), answers.cpu().tolist() ) ]
        else:
             # b,c0 # note: (x[:,1]==y[:,0]).all()
            if self.config.base == 'word':
                x,y = batch
                b,l = x.shape
                logits = self(x[:,-context:])
                preds = torch.argmax(logits[:,-1],dim=-1) # b
                true = y[:,-1]
                if batch_idx == logging_batch_idx:
                    rows = [[" ".join([self.rev[_c] for _c in _context]), self.rev[_pred], self.rev[_true].strip()] for _context, _pred, _true in zip(x[:,-context:].cpu().tolist(), preds.cpu().tolist(), true.cpu().tolist())]
            else: # e2e
                x, y, x_mask, y_mask = batch
                b,l,c = x.shape
                #logits = self(x[:,-context:], x_mask[:, -context:]) # b,context-1,c0,V
                #intermediates = []
                preds = torch.zeros((b,1),device=x.device,dtype=x.dtype)
                for i in range(c): # TODO: efficiently store output for the last word, just run last AR layer
                    logits = self(x[:,-context:], x_mask[:, -context:]) # b,context-1,c0,V
                    logits_ = logits[:, -1, i, :] / 1.0 # b,V
                    probs = F.softmax(logits_, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    preds = torch.cat((preds, idx_next),dim=1)
                    #idx = torch.cat((idx, idx_next), dim=1)
                    # check if idx_next matches original x[:,-1,i] for unit_acc. or keep aggregating, will retain old preds
                    x[:,-1,i] = idx_next.squeeze() # x[:,-1] is only used to initialize canvas with hints.
                    #intermediates.append(idx_next.squeeze().cpu().tolist())
                    #'''
                #preds = torch.stack(intermediates, dim=1) # b,c0
                #preds = torch.argmax(logits[:,-1],dim=-1) # b,c0
                preds = preds[:,1:]
                true = y[:,-2]
                if batch_idx == 0:
                    if self.config.base == 'char':
                        rows = [[" ".join(["".join([self.rev[_c] for _c in _word]).strip() for _word in _context]), "".join([self.rev[_c] for _c in _pred]).strip(), "".join([self.rev[_c] for _c in _true]).strip()] for _context, _pred, _true in zip(y[:,-(context+2):-2].cpu().tolist(), preds.cpu().tolist(), true.cpu().tolist())]
                    elif self.config.base in ['sub','byte']:
                        rows = [[" ".join([self.config.vocab.decode([_c for _c in _word if _c != 0]) for _word in _context]), self.config.vocab.decode([_c for _c in _pred if _c != 0]), self.config.vocab.decode([_c for _c in _true if _c != 0]).strip()] for _context, _pred, _true in zip(y[:,-(context+2):-2].cpu().tolist(), preds.cpu().tolist(), true.cpu().tolist())]
            corrects = (preds==true)
            acc_unit = (corrects*(true!=0).float()).sum() / (true!=0).float().sum() # sub/char TODO: exclude 0s
            if self.config.base == 'word':
                acc_word = acc_unit
            else:
                acc_word = corrects.all(dim=-1).float().mean()
        if batch_idx == logging_batch_idx:
            headers = ["context", "pred", "true"]
            #rows.insert(0, headers)
            self.logger.experiment.log_table("predictions.csv", [[f"\"{cell}\"" for cell in row] for row in rows], headers=headers)
        #rev = {k:v for v,k in self.config.vocab.items()}
        self.log("gpu", torch.cuda.memory_allocated() / (1024 ** 3), on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_unit', acc_unit, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('acc_word', acc_word, on_step=True, on_epoch=True, prog_bar=True, logger=True)
