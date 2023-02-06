from torch.utils.data import Dataset, DataLoader
import torch
import regex as re

import random, math, collections
import numpy as np

from transformers import AutoTokenizer

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = list(set(data))
        #chars.remove(' '); chars = [' '] + chars # index is 0
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

class eDataset(Dataset):
    def __init__(self, data, block_size, word_vocab_size=1000):
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r' +',r' ',text).strip()
        chars = list(set(text))
        chars.remove(' '); #chars = [' '] + chars # index is 0
        self.data = text.split(' ')
        words = list(set(self.data))
        self.maxlen = max(len(_) for _ in words) # max number of chars in a word

        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data), len(words)))
        #fwords = collections.Counter(self.data).most_common(word_vocab_size-1)  
        #print(f'Top {word_vocab_size-1} words cover {100*sum([_[1] for _ in fwords])/len(self.data):.2f}% of all words')
        #words = [_[0] for _ in fwords]

        self.ctoi = { ch:i for i,ch in enumerate(chars) }
        self.itoc = { i:ch for i,ch in enumerate(chars) }
        #self.wtoi = collections.defaultdict(lambda w:0)
        #self.itow = collections.defaultdict(lambda i:'UNK') 
        
        #for i,w in enumerate(words):
        #    self.wtoi[w] = i+1
        #    self.itow[i+1] = w
        self.block_size = block_size
        #self.wvocab_size = word_vocab_size
        self.cvocab_size = len(chars)

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]

        idxs = [[0]+[self.ctoi[_] for _ in word] for word in chunk]
        mask = torch.tensor([len(_)-1 for _ in x], dtype=torch.long)
        x = torch.tensor([_ + [0]*(self.maxlen+1-len(_)) for _ in idxs[:-1]], dtype=torch.long) # W,Lc
        y = torch.tensor([self.wtoi.get(_,0) for _ in chunk[1:]], dtype=torch.long) # W
        return x, y, mask

class eDataset_nat(Dataset):
    def __init__(self, data, block_size, word_vocab_size=None):
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r' +',r' ',text).strip()
        self.data = text.split(' ')
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        self.data = [_ if len(_)<=9 else "@" for _ in self.data] # removes 4.6% tokens
        self.bigram = self.get_bigram()
        print(f"{len(self.bigram)=}")
        words = list(set(self.data))
        self.maxlen = max(len(_) for _ in words) # max number of chars in a word
        print(f"{self.maxlen=}")

        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data), len(words)))
        self.ctoi = { ch:i for i,ch in enumerate(chars) }
        self.itoc = { i:ch for i,ch in enumerate(chars) }
        
        if word_vocab_size:
            fwords = collections.Counter(self.data).most_common(word_vocab_size-1)  
            print(f'Top {word_vocab_size-1} words cover {100*sum([_[1] for _ in fwords])/len(self.data):.2f}% of all words')
            words = [_[0] for _ in fwords]
            self.wtoi = collections.defaultdict(lambda w:0)
            self.itow = collections.defaultdict(lambda i:'UNK') 
            for i,w in enumerate(words):
                self.wtoi[w] = i+1
                self.itow[i+1] = w
            self.wtoi = dict(self.wtoi)
            self.itow = dict(self.itow)
            self.wvocab_size = word_vocab_size
        else:
            self.itow = None
            self.wtoi = None
            self.wvocab_size = None
        self.block_size = block_size
        self.cvocab_size = len(chars)

    def get_bigram(self):
        bigram = collections.defaultdict(lambda: 0)
        for w,f in collections.Counter(self.data).most_common():
            w += " "*(10-len(w))
            bigram[('<bos>',w[0])] += f
            for i,c in enumerate(w[:-1]):
                bigram[(c,w[i+1])] += f
        return dict(bigram)

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]

        #x = [[0]+[self.ctoi[_] for _ in word] for word in chunk[:-1]]
        
        idxs = [[self.ctoi[_] for _ in word] for word in chunk]
        #mask = torch.tensor([len(_)-1 for _ in x], dtype=torch.long)
        mask = [len(_) for _ in idxs]
        x = torch.tensor([_ + [0]*(self.maxlen+1-len(_)) for _ in idxs[:-1]], dtype=torch.long) # W,Lc
        x_mask = torch.tensor(mask[:-1], dtype=torch.long)
        if self.wvocab_size:
            y = torch.tensor([self.wtoi.get(_,0) for _ in chunk[1:]], dtype=torch.long) # W
            return x, y, x_mask
        else:
            y = torch.tensor([_ + [0]*(self.maxlen+1-len(_)) for _ in idxs[1:]], dtype=torch.long) # W,Lc
            y_mask = torch.tensor(mask[1:], dtype=torch.long)
            return x, y, x_mask, y_mask

class eDataset_char(Dataset):
    def __init__(self, data, block_size, **kwargs): # here block size is the number of chars
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r' +',r' ',text).strip()
        self.data = text.split(' ')
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        self.data = [_ if len(_)<=9 else "@" for _ in self.data] # removes 4.6% tokens
        words = list(set(self.data))
        self.maxlen = max(len(_) for _ in words) # max number of chars in a word
        print(f"{self.maxlen=}")
        self.data = ' '.join(self.data)

        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data), len(words)))
        self.ctoi = { ch:i for i,ch in enumerate(chars) }
        self.itoc = { i:ch for i,ch in enumerate(chars) }
        self.wtoi = None
        self.itow = None
        self.block_size = block_size
        self.cvocab_size = len(chars)

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        #idxs = [[self.ctoi[_] for _ in word] for word in chunk]
        idxs = [self.ctoi[_] for _ in chunk]
        x = torch.tensor(idxs[:-1], dtype=torch.long)
        y = torch.tensor(idxs[1:], dtype=torch.long)
        return x, y

class eDataset_sub(Dataset):
    def __init__(self, data, block_size, **kwargs): # here block size is the number of chars
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r' +',r' ',text).strip()
        self.data = text.split(' ')
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        self.data = [_ if len(_)<=9 else "@" for _ in self.data] # removes 4.6% tokens
        words = list(set(self.data))
        self.maxlen = max(len(_) for _ in words) # max number of chars in a word
        print(f"{self.maxlen=}")
        self.data = ' '.join(self.data)
        self.vocab = bpe
        #self.data = self.vocab.tokenize(' '.join(self.data)) # TOKENIZERS_PARALLELISM=false 

        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data), len(words)))
        
        self.ctoi = { ch:i for i,ch in enumerate(chars) }
        self.itoc = { i:ch for i,ch in enumerate(chars) }
        self.wtoi = None
        self.itow = None
        self.block_size = block_size
        self.cvocab_size = len(self.vocab)

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+5*self.block_size+1] # 4x chars to cast a wide net
        idxs = self.vocab(chunk, truncation=True, max_length=self.block_size+1)['input_ids']
        if len(idxs) != self.block_size + 1:
            return self.__getitem__(idx)
        #idxs = [[self.ctoi[_] for _ in word] for word in chunk]
        #idxs = [self.ctoi[_] for _ in chunk]
        #idxs = self.vocab.convert_tokens_to_ids(chunk)
        x = torch.tensor(idxs[:-1], dtype=torch.long)
        y = torch.tensor(idxs[1:], dtype=torch.long)
        return x, y

class myDataset(Dataset):
    def __init__(self, 
                 data, 
                 block_size, 
                 base='char', 
                 vocab_size=0, # if 0 then no limit else replace by UNK
                 do_e2e=False, 
                 vocab=None,
                 #maxlen=None,
                 ):
        text = data
        for remove in ['\n','<unk>','=', '@-@']:
            text = text.replace(remove,' ')
        text = re.sub(r"(-|{|}:|,|;|\.|\n|!|'|--|\?)",r' \1 ',text)
        text = re.sub(r'\b(?:[^\s]*[a-zA-Z])+[^\s]*\b', '', text)
        self.data = re.sub(r' +',r' ',text).strip()
        chars = list(set(text))
        chars.remove(' '); chars = [' '] + chars # index is 0
        words = list(set(self.data.split(' ')))
        print('data has %d characters, %d unique; %d words, %d unique' % (len(data), len(chars), len(self.data.split(' ')), len(words)))
        if do_e2e or base == 'word':
            self.data = self.data.split(' ')
        #self.vocab = [{},{}]    
        if base == 'char':
            self.vocab = { ch:i for i,ch in enumerate(chars) }
            self.rev = {k:v for v,k in self.vocab.items()}
        elif base == 'word':
            self.vocab = { w:i for i,w in enumerate(words) }
            self.rev = {k:v for v,k in self.vocab.items()}
        elif base == 'sub':
            #self.vocab = dict(bpe.vocab)
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
        if vocab:
            self.vocab = vocab
            if base in ['char','word']:
                self.rev = {k:v for v,k in self.vocab.items()}
            #self.maxlen = maxlen
        print(f"{self.maxlen=}")
        
        self.block_size = block_size
        self.do_e2e = do_e2e
        self.base = base
        
            

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx, chunk=None, recursion=0):
        k = 1
        if self.base == 'sub' and not self.do_e2e:
            k = 5
        if not chunk: # can give space-sep (for e2e) or raw text too.
            i = np.random.randint(0, len(self.data) - (self.block_size + 1)) 
            # we're actually going to "cheat" and pick a spot in the dataset at random
            chunk = self.data[i:round(i+k*self.block_size+1)]

        #x = [[0]+[self.ctoi[_] for _ in word] for word in chunk[:-1]]
        if self.do_e2e: # chunk has block_size words
            if self.base in ['sub','byte']: # TODO: 0 may not be padding here?
                idxs = [self.vocab(word, truncation=True, max_length=self.maxlen, add_special_tokens=False)['input_ids'] for word in chunk]
            else:    
                idxs = [[self.vocab[_] for _ in word] for word in chunk]
            #mask = torch.tensor([len(_)-1 for _ in x], dtype=torch.long)
            mask = [len(_) for _ in idxs]
            x = torch.tensor([_ + [0]*(self.maxlen-len(_)) for _ in idxs[:-1]], dtype=torch.long) # W,Lc
            x_mask = torch.tensor(mask[:-1], dtype=torch.long)
            y = torch.tensor([_ + [0]*(self.maxlen-len(_)) for _ in idxs[1:]], dtype=torch.long) # W,Lc
            y_mask = torch.tensor(mask[1:], dtype=torch.long)
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