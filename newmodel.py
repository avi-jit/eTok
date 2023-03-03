import math
import logging

import torch

import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics

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
        
        if base == 'sub':
            news = {v for k,v in vocab.vocab.items() if k.startswith('Ġ')} # these are word starter ids
            self.news = torch.tensor(list(news), dtype=torch.int64).unsqueeze(0)
        elif base in ['char','word']:
            self.rev = {k:v for v,k in vocab.items()}

        # end-to-end tokens
        if num_prefix > 0:
            assert base != 'word'
            encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_e2e_head, batch_first=True)
            self.wordenc = nn.TransformerEncoder(encoder_layer, num_layers=n_e2e_layer)
            canvas_size = canvas_size
            self.word_pe = nn.Parameter(torch.zeros(1, block_size*num_prefix, n_embd))
        else:
            canvas_size = block_size
        
        # input embedding stem
        self.in_emb = nn.Embedding(len(vocab) + num_prefix, n_embd)
        self.in_pe = nn.Parameter(torch.zeros(1, canvas_size + num_prefix, n_embd)) # 21 will do
        #self.word_pe = nn.Parameter(torch.zeros(1, block_size, n_embd*num_prefix))
        self.drop = nn.Dropout(embd_pdrop)
        
        # transformer
        #self.config.n_embd = n_embd * num_prefix
        self.config.block_size = block_size * max(num_prefix,1) # for blocks - shouldn't it be n_head?
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        self.config.block_size = block_size # back to normal; to save this in ckpt

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        if num_prefix > 0:
            self.config.block_size = canvas_size + num_prefix # for blocks
            self.decoder_blocks = nn.Sequential(*[Block(self.config) for _ in range(n_e2e_layer)])
            self.config.block_size = block_size # back to normal; to save this in ckpt
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

    def _make_mask(self, b, t, c):
        mask = torch.zeros((t*c, t*c))
        for i in range(t*c):
            len = (i+1)%c if (i+1)%c else c
            start = int(i/c)*c
            for j in range(start, start+len):
                mask[i][j] = 1
        mask = torch.stack([mask for _ in range(b)])
        return mask
    
    def forward(self, idx, mask=None, eval=False):
        p = self.config.num_prefix
        assert idx.size()[1] <= self.config.block_size, "Cannot forward, model block size is exhausted."
        if p > 0:
            b, t, c0 = idx.size(); # mask is b,t. needs to be (4096 = 8 heads * 512 = 4 b * 128 t, 21, 21) or (21, 21).
            # prefix the idx
            mask += self.config.num_prefix
            prefix = torch.tensor(range(c0,c0+p,), device=idx.device)
            x = torch.cat((prefix.unsqueeze(0).unsqueeze(0).repeat(b,t,1),idx),-1)
            b, t, c = x.size()
            
#             token_embeddings = self.in_emb(x) # each index maps to a (learnable) vector
#             in_pe = self.in_pe[:, :c, :].unsqueeze(1) # each position maps to a (learnable) vector
#             token_embeddings = self.drop(token_embeddings + in_pe) # B,t,D. so far 1 word 1 embedding.

            mask = mask.reshape(-1) # b*t
            mask = torch.arange(c, device=mask.device).expand(len(mask), c) > mask.unsqueeze(1) # b*t,c
            mask0 = torch.repeat_interleave(torch.einsum('bi,bj->bij', mask, mask), self.config.n_e2e_head, dim=0))
        
            #out = self.wordenc(token_embeddings.reshape(b*t,c,-1), mask=self.make_mask(b, t, c))
            
            query = out[:,:p,:] # first few denote CLS per word. 2560,4,512 = b*t,p,d. earlier was b*t,d
            
            word_pe = self.word_pe[:, :t*p, :] # 1,128,512 = 1,t,d
            in_emb = self.drop(query.reshape(b,t*p,-1) + word_pe) # b,t*p,d i.e. all p prefixes are separate tokens. way costlier due to O(n2) attn        
        else:
            b, t = idx.size(); 
            in_emb = self.in_emb(idx) # each index maps to a (learnable) vector        
        
        net = self.blocks(in_emb) # b,t*p,d - todo: try variant where each word gets 1 token p*d
        net = self.ln_f(net)

        if p > 0:
            if eval:
                return net.reshape(b,t,p,-1)[:,-2,:,:] # b,t-1,p,d so first t-1 outputs
            # net is (b,t*p,d). use these to replace first p of chars: (b,t,c=p+c0,d)
            canvas = token_embeddings[:,1:,:,:] # b,t-1,c,d - stores hints 1 onwards for AR training
            canvas[:,:,:p,:] = net.reshape(b,t,p,-1)[:,:-1,:,:] # b,t-1,p,d so first t-1 outputs
            tokens_out = self.drop(canvas + in_pe)
            out = self.decoder_blocks(tokens_out.reshape(b*(t-1),c,-1))
            logits = self.head(out.reshape(b,t-1,c,-1)[:,:,p-1:-1,:]) # b,t-1,c0,Vc
            # from p-1 to c-1, i.e., BOS (no input) to when last but one char is also input (out should be last) 
        else:
            logits = self.head(net.reshape(b,t,-1)) # b,t*p,d -> b,t,Vw
        return logits

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
            
'''    def validation_epoch_end(self, outputs) -> None:
        if outputs==[]:
            return
        acc_c0 = torch.cat([_[0] for _ in outputs]); acc_c = torch.cat([_[1] for _ in outputs]); acc_w0 = torch.cat([_[2] for _ in outputs]); 
        mask = torch.cat([_[3] for _ in outputs]);
        self.log("val_acc_char+0", acc_c0.sum()/acc_c0.numel(), prog_bar=True) # overestimate acc because 0 is easy
        self.log("val_acc_char", acc_c.sum()/mask.sum(), prog_bar=True)
        self.log("val_acc_word+0", acc_w0.sum()/acc_w0.numel(), prog_bar=True) # fair acc because 0 is needed'''

















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

