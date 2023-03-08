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
                 cls_token='@'
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
        self.cls_token = cls_token
        
        empt = {v for k, v in vocab.items() if k == ' '} # ' ' is paddingc
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
            encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_e2e_head, batch_first=True)
            self.wordenc = nn.TransformerEncoder(encoder_layer, num_layers=n_e2e_layer)
            canvas_size = canvas_size
            self.word_pe = nn.Parameter(torch.zeros(1, block_size*num_prefix, n_embd))
        else:
            canvas_size = block_size
        
        # input embedding stem
        self.in_emb = nn.Embedding(len(vocab) + num_prefix, n_embd)
        self.in_pe = nn.Parameter(torch.zeros(1, canvas_size + num_prefix, n_embd))
        self.seq_pe = nn.Parameter(torch.zeros(1, block_size))
        #self.drop = nn.Dropout(embd_pdrop)

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

    def _add_cls(self, idx):
        B,t = idx.size()
        idx_with_cls = []
        for i in range(B):
            head_indx_list = torch.where(idx[i].unsqueeze(1) == self.news)[0]
            now_idx_with_cls = tolist(idx[i])
            for head_indx in reversed(head_indx_list):
                for offset in range(self.config.num_prefix):
                    now_idx_with_cls.insert(head_indx, self.vocab_length+self.config.num_prefix-1-offset)
            idx_with_cls.append(now_idx_with_cls)
         return torch.nn.utils.rnn.pad_sequence(idx_with_cls, batch_first = True)
    
    def _get_cls_indx(self, idx):
        cls_indx_list = []
        for seq in idx:
            cls_indx_list.append(torch.where(seq.unsqueeze(1) == self.news)[0])
        cls_indx_list = [x - self.config.num_prefix for x in cls_indx_list]
        return cls_indx_list

    def _make_mask(self, idx, cls_indx_list):
        for cls_indx in cls_indx_list:
            mask = torch.zeros((idx.size()[1], idx.size()[1]))
            indx_list_len, = cls_indx.size()
            for i in range(indx_list_len):
                now = cls_indx[i]
                nxt = len(cls_indx) 
                if(not (i == indx_list_len-1)):
                    nxt = cls_indx[i+1]
                for tri_i in range(now, nxt):
                    for tri_j in range(now, now+1+(tri_i-now)):
                        mask[tri_i][tri_j] = 1
            masks.append(mask)
        masks = torch.stack(masks)
        return masks # [B, t, t]
    
    def _extract_cls(self, out, cls_indx_list): #(!) cls_out is padded
        cls_out = []
        B,_,_ = out.size()
        for i in range(B):
            cls_out.append(torch.tensor([ out[i][i:i+self.config.num_prefix, :] for i in cls_indx_list[i]]))
        cls_out = torch.nn.utils.rnn.pad_sequence(cls_out, batch_first = True)
        return cls_out

    def _get_canvas(self, token_embeddings, net, cls_indx_list):
        canvas = []
        B, t, _ = token_embeddings.size()

        for i in range(B):
            for token_row in token_embeddings:
                canvas.append(token_row[cls_indx_list[i][1]:, :])     # [t_from2, c] remove the first word

        for i in range(B):
            for cls_indx in range(cls_indx_list[:-1]) :
                cls_shift = cls_indx_list[:-1][cls_indx]
                canvas[i, cls_shift:cls_shift+self.config.num_prefix,:] = net[i, cls_shift:cls_shift+self.config.num_prefix,:]
        
        canvas = torch.nn.utils.rnn.pad_sequence(canvas, batch_first = True)
        return canvas

    def _get_out_word(self, out, cls_indx_list):
        B, t, _ = out.size()
        out_word = []
        for i in range(B):
            cls_shift = cls_indx_list[i][1]
            now_word = []
            for cls_seq_indx in range(len(cls_indx_list[i][:-1])):
                pre_cls = cls_indx_list[i][:-1][cls_seq_indx]
                if(not cls_seq_indx == (len(cls_indx_list[i][:-1])-1) ):
                    nxt_cls = cls_indx_list[i][:-1][cls_seq_indx+1]
                else:
                    nxt_cls = t
                    now_word.append(torch.squeeze(out[i,pre_cls-cls_shift+self.config.num_prefix-1:nxt_cls,:])) #(word_len, embd)
            now_word = torch.stack(now_word) 
            out_word.append(now_word)
        out_word = torch.nn.utils.rnn.pad_sequence(out_word, batch_first = True)
        return out_word #[B, t, embd]

#(!) positional embedding not changed
    def forward(self, idx, mask=None, eval=False): # based on assumption, prefix is added before forward function
        p = self.config.num_prefix
        assert idx.size()[1] <= self.config.block_size, "Cannot forward, model block size is exhausted."
        
        # init encoding
        if p > 0:
            idx = self._add_cls(idx)
            B, t = idx.size()
            token_embeddings = self.in_emb(idx)
            in_pe = self.in_pe[:, :t, :]
            token_embeddings = self.drop(token_embeddings + in_pe) # [B, t, embd]
            cls_indx_list = self._get_cls_indx(idx)
            e2e_masks = self._make_mask(idx, cls_indx_list) #[B, t, t] (?) don't know how to do in batch

            out = self.wordenc(token_embeddings, e2e_masks) #[B, t, embd]
            query = self._extract_cls(out, cls_indx_list) # (!) query is a list of tensor [B, t_cls_pad, embd]
            
            _, tq, _ = query.size() #[B, t_cls_pad, embd]
            word_pe = self.word_pe[:, tq, :]
            in_emb = self.drop(query + word_pe)

        else:
            b, t = idx.size()
            in_pe = self.in_pe[:, :t]
            in_emb = self.wordenc(token_embeddings)

        
        net = self.blocks(in_emb)
        net = self.ln_f(net)

        if p > 0:
            if eval:
                return net #(?) don't really understand what does original reshape mean
            
            canvas = self._get_canvas(token_embeddings, net, cls_indx_list)
            B, t_canvas, _ = canvas.size()
            canvas_pe = self.in_pe[:, :t_canvas, :] #(!) change positional embeddings!
            tokens_out = self.drop(canvas+canvas_pe)
            out = self.decoder_blocks(tokens_out)

            logits = self.head(self._get_out_word(out))
        else:
            logits = self.head(net.reshape(b, t, -1))
        return logits

#(!) training, generate, validate not changed
    def training_step(self, batch, batch_idx, eval=False):
        if self.config.num_prefix == 0:
            x, y = batch
            logits = self(x)
        else:
            x, y, x_mask, y_mask = batch
            logits = self(x, x_mask) # b,t,wvocab

        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if self.config.num_prefix > 0:
                loss = F.cross_entropy(logits.transpose(1,2).transpose(1,3), y[:,:-1,:]) # logits (b,t-1,c0,Vc). y (b,t,c0)
            else:
                loss = F.cross_entropy(logits.transpose(1,2), y)
        if not eval:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("gpu", torch.cuda.memory_allocated() / (1024 ** 3), on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
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
                    '''
                    logits = self(x[:,-context:], x_mask[:, -context:], eval=True) # b,c0,V
                    canvas = token_embeddings[:,1:,:,:] # b,t-1,c,d
                    canvas[:,:,:p,:] = net.reshape(b,t,p,-1)[:,:-1,:,:] # b,t-1,p,d so first t-1 outputs
                    tokens_out = self.drop(canvas + in_pe)
                    out = self.decoder_blocks(tokens_out.reshape(b*(t-1),c,-1))
                    logits = self.head(out.reshape(b,t-1,c,-1)[:,:,p-1:-1,:]) # b,t-1,c0,Vc
                    '''
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

class TransformerDecoder(torch.nn.Module):
    """  decoder only GPT-like language model, with a context size of block_size """
    def __init__(self,
                 num_heads=4,
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
                 **kwargs,
                 ):
        super().__init__()
        
        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams
        self.config.num_prefix=''

        # input embedding stem
        self.char_pe = nn.Parameter(torch.zeros(1, block_size, n_embd)) # 21 will do
        self.drop = nn.Dropout(embd_pdrop)
        
        # transformer
        self.config.block_size = block_size # for blocks
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        self.config.block_size = block_size # back to normal; to save this in ckpt

        # decoder head
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
        b, t, d = x.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        char_pe = self.char_pe[:, :t, :] # each position maps to a (learnable) vector
        chars = self.drop(x + char_pe) # B,t,D. so far 1 word 1 embedding.

        net = self.blocks(chars) # b,t,d
        net = self.ln_f(net)
        return net
