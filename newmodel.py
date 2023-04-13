import math
import logging

import torch

import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import csv

from mingpt.model import Block

logger = logging.getLogger(__name__)

class SelfAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_e2e_head, batch_first = True)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )
    
    def forward(self, x, mask=None):
        x = self.ln1(x)
        x = x + self.attn(x, x, x, attn_mask = mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttentionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(config) for _ in range(config.n_e2e_layer)])
        
    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return x

class myGPT(pl.LightningModule):
    """  end-to-end tokenized full GPT language model, with a context size of block_size """
    def __init__(self,
                 #e2e_vocab_size=100,
                 num_prefix=1, # 0 for no compression
                 cls_token='@',
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
                 lang = 'en',
                 dataset = 'shakespare',
                 batch_size = 2,
                 save_to_val_csv=True,
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature

        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams
        self.vocab_length = len(vocab)
        self.cls_token = cls_token

        if base == 'sub':
            news = {v for k,v in vocab.get_vocab().items() if k.startswith('Ġ')} # (?) shouldn't they all be first token for each c
            self.news = torch.tensor(list(news), dtype=torch.int64).unsqueeze(0)
        elif base in ['char','word']:
            self.rev = {k:v for v,k in vocab.items()}

        # end-to-end tokens
        if num_prefix > 0: # only prefix = 1 is considered
            assert base != 'word'
            encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_e2e_head, batch_first=True)
            self.wordenc = nn.TransformerEncoder(encoder_layer, num_layers=n_e2e_layer)
            self.word_pe = nn.Parameter(torch.zeros(block_size, n_embd))
        else:
            canvas_size = block_size
        
        # input embedding stem
        self.in_emb = nn.Embedding(len(vocab), n_embd)
        self.in_pe = nn.Parameter(torch.zeros(block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer (!) check whether blocks is changed
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        if num_prefix > 0:
            self.decoder_blocks = SelfAttentionDecoder(self.config)
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

# get cls index (padding indexs are 0 at the end) (!) assumption: each input sequence has same number of subword tokens, different num of words
    def _get_cls_indx(self, mask):
        # mask = [B, t]
        cum = torch.cumsum(mask, dim=-1) # cum = [B]
        token_len = torch.sum(mask[0])
        word_lens = torch.clone(mask)
        cls_indx = cum - word_lens
        cls_indx = torch.where(cls_indx < token_len, cls_indx, 0)
        return cls_indx

# make local attn masks
    def _make_mask(self, mask): 
        p = torch.nn.utils.rnn.pad_sequence(mask, batch_first = True) #(!) why there is a pad seq
        b, l = p.shape # what is p 
        n = sum(p[0]) # not finished
        n = n.item()
        #print(torch.cumsum(p,-1))
        #print(torch.cumsum(p,-1).reshape(-1).shape)
        #print(torch.cumsum(p,-1).reshape(-1).repeat_interleave(l).shape)
        #print(b, l, n)
        y_ = torch.cumsum(p,-1).reshape(-1).repeat_interleave(p.reshape(-1)).reshape((b,n))
        x_ = y_ - p.reshape(-1).repeat_interleave(p.reshape(-1)).reshape((b,n))
        coords_ = torch.arange(0, n, dtype=int)
        coords = torch.cartesian_prod(coords_, coords_)
        mins = torch.min(coords, dim=1).values
        maxs = torch.max(coords, dim=1).values
        x_ = x_.cuda()
        y_ = y_.cuda()
        mins = mins.cuda()
        maxs = maxs.cuda()
        masks = (x_.repeat_interleave(n, dim=-1) <= mins) * (maxs < y_.repeat_interleave(n, dim=-1))
        attn_masks = torch.tril(masks.reshape((b,n,n))).bool()
        return attn_masks

# (!) need change 
    def _extract_cls(self, out, cls_indx): #(!) cls_out is padded
        # out = [B, t, embd]
        # cls_indx = [B, t] #t with pad
        cls_out = []
        B, t, embd = out.shape
        for i in range(B):
            # if actual cls -> extract 
            # if padded cls -> padding 0
            if(cls_indx[i].eq(0).all()):
                cls_out.append(torch.zeros(1, embd))
                continue
            end_idx = (cls_indx[i] != 0).nonzero(as_tuple=True)[0][-1]
            end_idx = end_idx.item()
            cls_indx_seq_truncated = cls_indx[i, :end_idx+1]
            cls_out.append(torch.tensor(out[i][cls_indx_seq_truncated]))
        cls_out = torch.nn.utils.rnn.pad_sequence(cls_out, batch_first = True)
        return cls_out
            
        
    def _get_canvas(self, out, net, cls_indx, mask):
        B, t, _ = out.size()
        #out_new = torch.zeros((B, t, _), device=out.device)
        for i in range(B):
            if(cls_indx[i].eq(0).all()): #(!) why continue
                continue
            end_idx = (cls_indx[i] != 0).nonzero().max()
            end_idx = end_idx.item()
            cls_indx_seq_truncated = cls_indx[i, :end_idx+1] # truncated cls indices
            cls_indx_delete_last = torch.roll(cls_indx_seq_truncated, shifts=-1, dims=0)
            cls_indx_delete_last = cls_indx_delete_last[:-1]
            out[i][cls_indx_delete_last] = net[i, :end_idx] 
            out[i][:cls_indx_delete_last[0]][:] = 0
            #out_new[i] = torch.roll(out[i], shifts=-cls_indx_delete_first[0], dims=0)
            mask[i][:cls_indx_delete_last[0], :] = 0
            mask[i][:, :cls_indx_delete_last[0]] = 0
            
        return out, mask

#(!) positional embedding not changed
    def forward(self, idx, mask=None, eval=False): # based on assumption, prefix is added before forward function
        p = self.config.num_prefix
        assert idx.size()[1] <= self.config.block_size, "Cannot forward, model block size is exhausted."
        
        # init encoding
        if p > 0:
            #B, t = idx.size()
            token_embeddings = self.in_emb(idx)
            B, t, embd = token_embeddings.size()
            in_pe = self.in_pe[:t, :]
            token_embeddings = self.drop(token_embeddings + in_pe) # [B, t, embd]
            cls_indx = self._get_cls_indx(mask) 
            #print(mask.shape)
            #print('mask:')
            #print(mask)
            e2e_masks = self._make_mask(mask) #[B, t, t] (?) don't know how to do in batc
            #print(e2e_masks)
            #print('mask end')
            #print(e2e_masks.shape)
            #print(self.config.n_e2e_head)
            #print(torch.repeat_interleave(e2e_masks, self.config.n_e2e_head, dim=0).shape)
            out = self.wordenc(token_embeddings, mask = torch.repeat_interleave(e2e_masks, self.config.n_e2e_head, dim=0))
            #out = self.wordenc(token_embeddings, e2e_masks.bool()) #[B, t, embd]
            query = self._extract_cls(out, cls_indx) # query = [B, cls_max, embd]
            
            _, tq, _ = query.size() #[B, cls_max, embd]
            word_pe = self.word_pe[:tq, :]
            in_embd = self.drop(query + word_pe)

        else:
            b, t = idx.size()
            in_pe = self.in_pe[:, :t]
            in_embd = self.wordenc(token_embeddings)

        
        net = self.blocks(in_embd) #in_emb is the embeddingsafter local attn
        net = self.ln_f(net)

        if p > 0:
            if eval:
                return net #(?) don't really understand what does original reshape mean
            #print(out)
            canvas, e2e_masks = self._get_canvas(token_embeddings, net, cls_indx, e2e_masks)
            #print(out)
            #print("!")
            B, t_canvas, _ = canvas.shape
            #canvas_pe = self.out_pe[:, :t_canvas, :] 
            tokens_out = self.drop(canvas+in_pe) #[B, t, embd]
            #print(e2e_masks) 
            result = self.decoder_blocks(tokens_out, mask=torch.repeat_interleave(e2e_masks, self.config.n_e2e_head, dim=0))
            #print(out)
            #print(out)
            logits = self.head(result)
        else:
            logits = self.head(net.reshape(b, t, -1))
        return logits

    def delete_first_word(self, logits, y, mask):
        B, t = y.shape
        B, t, len_voc = logits.shape
        new_logits = torch.zeros(1, len_voc, device=logits.device)
        new_y = torch.zeros(1, device=y.device, dtype=torch.long)
        for i in range(B):
            now_y = y[i][mask[i][0]:]
            now_y = torch.squeeze(now_y)
            new_y = torch.cat((new_y, now_y))
            now_logits = logits[i][mask[i][0]:][:]
            now_logits = torch.squeeze(now_logits)
            #print(now_logits.shape)
            #print(new_logits.shape)
            new_logits = torch.cat((new_logits,now_logits), 0)
        new_logits = new_logits[1:]
        new_y = new_y[1:]
        #print(new_logits)
        #print(new_y)
        return new_logits, new_y
            

#(!) training, generate, validate not changed
    def training_step(self, batch, batch_idx, eval=False):
        if self.config.num_prefix == 0:
            x, y = batch
            logits = self(x)
        else:
            x, y, x_mask, y_mask = batch
            #print('!')
            #print(x_mask)
            #print('!!')
            logits = self(x, x_mask) # b,t,wvocab

# logits = [B, t, vocab]
        # if we are given some desired targets also calculate the loss
        loss = None
        if y is not None:
            #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            if self.config.num_prefix > 0:
                #logits = logits[:, :-1, :] #
                B, t, vocab_len = logits.shape
                #logits = torch.reshape(logits, (-1, vocab_len))
                logits, y = self.delete_first_word(logits, y, x_mask)
                #print(logits)
                #print(y)
                loss = F.cross_entropy(logits, y)
            else:
                loss = F.cross_entropy(logits.transpose(1,2), y)
        if not eval:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("gpu", torch.cuda.memory_allocated() / (1024 ** 3), on_epoch=True, prog_bar=True, logger=True)
        return loss

# generate for e2e subword -> only modified the idx_cond index
    @torch.no_grad()
    def generate(self, idx, masks, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        B, t = idx.shape
        for _ in range(max_new_tokens):
            idx_cond = idx
            #print(idx_cond)
            logits = self(idx_cond, masks)
            logits = logits[:, -1, :] / temperature
            # (?) how to use top_k in probs
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            #print(logits)
            probs = F.softmax(logits, dim=-1)
            #print(probs)
            if do_sample:
                #idx_next = torch.multinomial(probs, k=1, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            idx = torch.cat((idx, idx_next), dim=1)

            for i in range(B):
                if(masks[i].eq(0).all()):
                    continue
                last_nonzero_indx = masks[i].nonzero().max().item()
                if(idx[i][-1] == self.config.vocab[self.cls_token]):
                    masks[i][last_nonzero_indx+1] = 1
                else:
                    masks[i][last_nonzero_indx] += 1

            
        return idx


    def validation_step(self, batch, batch_idx):
        logging_batch_interval = 3
        save_to_csv = self.config.save_to_val_csv
        #self.log('val_loss', self.training_step(batch, batch_idx, eval=True), on_step=False, on_epoch=True, logger=True)
        context= 90
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
                x, y, _, _ = batch
                b, t = x.shape
                inputs = torch.zeros((b, context), dtype=x.dtype, device=x.device)
                answers = torch.zeros((b, max_new_tokens), dtype=x.dtype, device=x.device)
                lens = torch.zeros(b, dtype=x.dtype, device=x.device)
                masks = torch.zeros((b, context), dtype=torch.long, device=x.device) #(!)
                for i, row in enumerate(x):
                    if self.config.base == 'char':
                        cls_heads = torch.where(row == self.config.vocab[self.cls_token])[0]
                        last_word_beg, last_word_end = cls_heads[-2], cls_heads[-1]
                        inputs[i] = row[last_word_beg - context+1:last_word_beg+1]
                        inputs[i][0] = self.config.vocab[self.cls_token] # such a bad hypothesis
                        
                        cls_heads = torch.where(inputs[i] == self.config.vocab[self.cls_token])[0]
                        cls_heads_shifted = torch.roll(cls_heads, shifts=-1, dims=0)
                        cls_heads_shifted[-1] = context
                        mask = cls_heads_shifted - cls_heads
                        #print(cls_heads_shifted)
                        #print(cls_heads)
                        x_mask = torch.tensor(mask, dtype=torch.long)
                        x_mask = torch.nn.functional.pad(x_mask, (0, context - (x_mask.shape)[-1]), mode='constant', value=0)
                        masks[i] = x_mask #pad to block_size (!)

                        now_len = last_word_end - last_word_beg -1

                        answers[i, :now_len] = row[last_word_beg+1:last_word_end]
                        lens[i] = now_len
                # based on how answer is defined, need to change answer
                #print('!!')
                #print(masks)
                #print('!!!')
                out = self.generate(inputs,masks,  max_new_tokens=max_new_tokens, temperature=1.0, do_sample=True, top_k=None)
                old_out = out.detach().clone()
                out = out[:, context:]
                mask = torch.arange(max_new_tokens, device=x.device).expand(len(lens), max_new_tokens) < lens.unsqueeze(1)
                corrects = (out==answers)*mask
                acc_unit = corrects.sum()/mask.sum()
                acc_word = (corrects.float() + (1- mask.float())).bool().all(dim=-1).float().mean()
                if batch_idx % logging_batch_interval == 0 and save_to_csv:
                    #if self.config.base == 'sub':
                    values = list(self.config.vocab.values())
                    keys = list(self.config.vocab.keys())
                    rows = [[[keys[values.index(_c)] for _c in seq if _c!=0], [keys[values.index(_c)] for _c in _pred_full if _c!=0], [keys[values.index(_c)] for _c in _pred if _c!=0], [keys[values.index(_c)] for _c in _true if _c!=0]] for seq, _pred_full, _pred, _true in zip(inputs.cpu().tolist(), old_out.cpu().tolist(), out.cpu().tolist(), answers.cpu().tolist())]
         
        if batch_idx % logging_batch_interval == 0 and save_to_csv:
            with open(f"val_output_{self.config.dataset}_{self.config.lang}_{self.config.base}_{'no-e2e' if self.config.num_prefix == 0 else 'e2e'}_{self.config.learning_rate}_{self.config.num_prefix}_{self.config.batch_size}.csv", 'a+', newline='') as val_csv:
                writer = csv.writer(val_csv, delimiter=',')
                for row in rows:
                    writer.writerow(row)
            print(f"val_output_{self.config.dataset}_{self.config.lang}_{self.config.base}_{'no-e2e' if self.config.num_prefix == 0 else 'e2e'}_{self.config.learning_rate}_{self.config.num_prefix}_{self.config.batch_size}.csv")
            #rows.insert(0, headers)
            #self.logger.experiment.log_table("predictions.csv", [[f"\"{cell}\"" for cell in row] for row in rows], headers=headers)
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
