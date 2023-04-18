from torch.utils.data import Dataset, DataLoader
import torch
import regex as re

import random, math, collections
import numpy as np
from itertools import chain
from transformers import AutoTokenizer

class myDataset(Dataset):
    def __init__(self, 
                 data, 
                 block_size, 
                 cls_token = '@',
                 base='char', 
                 vocab_size=0, # if 0 then no limit else replace by UNK
                 do_e2e=False, 
                 vocab=None,
                 #maxlen=None,
                 ):
        self.cls_token = cls_token
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(-|{|}:|,|;|\.|\n|!|'|--|:|\?)",r' ',text) #(?) why originally different
        text = re.sub(r' +',r' ',text)
        text = re.sub('\s{2,}', ' ', text)
        self.data = text
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        words = list(set(self.data.split(' ')))
        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data.split(' ')), len(words)))  
        if base == 'char':
            self.vocab = { ch:i for i,ch in enumerate(chars) }
            self.rev = {k:v for v,k in self.vocab.get_vocab().items()}
        elif base == 'word':
            self.vocab = { w:i for i,w in enumerate(words) }
            self.rev = {k:v for v,k in self.vocab.get_vocab().items()}
        elif base == 'sub':
            self.data = self.data.split(' ')
            self.vocab = AutoTokenizer.from_pretrained("gpt2")
        elif base == 'byte':
            self.vocab = AutoTokenizer.from_pretrained("google/byt5-small")  
        if vocab_size == 0:
            if base == 'char':
                vocab_size = len(chars)
                self.maxlen = max(len(_) for _ in words) # max number of chars in a word
            elif base == 'word':
                vocab_size = len(words)
                self.maxlen = 10
            elif base in ['sub','byte']:
                vocab_size = len(self.vocab)
                self.maxlen = max(len(self.vocab.tokenize(_)) for _ in words) # max number of subwords in a word
        else:
            raise NotImplemented # TODO: allow curbing vocab size with UNK
            
        if base == 'word':
            self.data = self.data.split(' ')
        if do_e2e and base == 'sub':
            #self.data = list(chain(*[self.vocab(word, truncation=True, max_length=self.maxlen, add_special_tokens=False)['input_ids'] for word in self.data.split(' ')])) #(!) pre tokenization
            self.data = list(self.data.split(' '))
            
        if vocab:
            self.vocab = vocab
            if base in ['char','word']:
                self.rev = {k:v for v,k in self.vocab.get_vocab().items()}
        print(f"{self.maxlen=}")
        
        self.block_size = block_size
        self.do_e2e = do_e2e
        self.base = base
        
            
    def __idx_mask__(self, idxs):
            cls_heads = torch.where(idxs == self.vocab(self.cls_token)['input_ids'][0])[0]            
            cls_heads_shifted = torch.roll(cls_heads, shifts=-1, dims=0)
            cls_heads_shifted[-1] = idxs.size()[-1]
            mask = cls_heads_shifted - cls_heads
            return torch.tensor(mask, dtype=torch.long)
            
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __get_mask__(self, idx, mask, seq_len):
        mask = mask.detach().clone()
        mask = torch.cumsum(mask, 0)
        mask = torch.where(mask < seq_len, mask, 0)
        end_idx = (mask!=0).nonzero(as_tuple=True)[0][-1]
        last_sum = mask[end_idx]
        add_value = seq_len - last_sum
        mask_shifted = torch.roll(mask, shifts=1, dims=0)
        mask_shifted[0] = 0
        mask_shifted[end_idx+1] = 0
        mask = mask - mask_shifted
        mask[end_idx+1] = add_value
        return mask
    
    def __getitem__(self, idx, chunk=None, recursion=0):
        k = 1
        word_shifter = 0
        if self.base == 'sub' and not self.do_e2e:
            k = 5
        if not chunk: # can give space-sep (for e2e) or raw text too.
            i = np.random.randint(0, len(self.data) - (k*self.block_size + 1)) 
            # we're actually going to "cheat" and pick a spot in the dataset at random
            chunk = self.data[i:i+k*self.block_size+2] # block_size ~ word size (!) +1 need to stay!!!!
        
        if self.do_e2e: # chunk has block_size words
            if self.base in ['sub','byte']: # TODO: 0 may not be padding here?
                idxs = []
            else:    
                idxs = [[self.vocab[_] for _ in word] for word in chunk]
            #mask = torch.tensor([len(_)-1 for _ in x], dtype=torch.long)
    
            mask = []
            for word in chunk:
              now_tokens = self.vocab(word, truncation=True, add_special_tokens=False)['input_ids']
              idxs += now_tokens
              mask.append(len(now_tokens))

            x = idxs.detach().clone()
            x = x[:seq_len]
            y = idxs.detach().clone()
            y = y[mask[0]:seq_len]
            mask = torch.tensor(mask)
            seq_len = self.block_size
            x_mask = self.__get_mask__(x, mask, seq_len)
            y_mask = x_mask.detach().clone()
          
            y = torch.nn.functional.pad(y, (self.block_size - (y.shape)[-1], 0), mode='constant', value=0)
            x_mask = torch.nn.functional.pad(x_mask, (0, self.block_size - (x_mask.shape)[-1]), mode='constant', value=0)
            y_mask = torch.nn.functional.pad(y_mask, (0, self.block_size - (y_mask.shape)[-1]), mode='constant', value=0)
            
            return x, y, x_mask, y_mask
        else: # chunk has block_size chars/bytes/subwords
            if self.base == 'word':
                idxs = [self.vocab[_] for _ in chunk]
            elif self.base in ['sub','byte']:
                idxs = self.vocab(chunk, truncation=True, max_length=self.block_size+1, add_special_tokens=False)['input_ids']
                if len(idxs) != self.block_size + 1:
                    if recursion > 5:
                        raise NotImplemented
                    return self.__getitem__(idx, recursion=recursion+1)
            elif self.base == 'char':
                idxs = [self.vocab[_] for _ in chunk]
            x = torch.tensor(idxs[:-1], dtype=torch.long)
            y = torch.tensor(idxs[1:], dtype=torch.long)
            return x, y
