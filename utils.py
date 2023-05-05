from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch import tensor


import nltk
# import tempfile
#
# nltk.download('punkt', download_dir=tempfile.gettempdir())
# nltk.download('averaged_perceptron_tagger', download_dir=tempfile.gettempdir())
# nltk.download('tagsets', download_dir=tempfile.gettempdir())
# nltk.data.path.append(tempfile.gettempdir())

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')

from nltk.data import load
paras = load('taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle')

from nltk import pos_tag

ss = 'NN NNS NNP NNPS VB VBD VBG VBN VBP VBZ JJ JJR JJS RBR BR RBS WDT WP WP$ PRP PRP$ DT CD UH SYM FW  LS'

selected_tags = ss.split()
pos_dict = {}
i = 0
for pos in selected_tags:
    pos_dict[pos] = i
    i += 1
n = len(pos_dict)

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WiCDataset(Dataset):

    def __init__(self, mode='train', root_dir='./WiC_dataset/'):

        self.root_dir = root_dir
        self.mode = mode

        file_name = root_dir + self.mode + '/'+ self.mode+'.data.txt'
        with open(file_name,'r', encoding = 'cp850') as file:    
            self.data = file.readlines()
           
        file_name = root_dir + self.mode + '/'+ self.mode +'.gold.txt'
        with open(file_name,'r', encoding = 'cp850') as file:
            self.labels = file.readlines()

    def __len__(self):
        return len(self.labels)
        

    def __getitem__(self, idx):

        line = self.data[idx]
        sample = {}
        parts = line.replace("\n","\t").strip().split("\t")
        sample['word'] = parts[0]

        if parts[1] == "F":
            sample['label'] = False
        else:
            sample['label'] = True

        sample['sentence1'] = parts[3]
        sample['sentence2'] = parts[4]

        idxs = parts[2].split('-')
        sample['idx1'] = idxs[0]
        sample['idx2'] = idxs[1]


        line = self.labels[idx]
        if line.split()[0] == 'F':
            sample['label'] = False
        else:
            sample['label'] = True

        return sample

    def get_vocab(self):  
        vocab = {"<UNK>":0}
        for line in self.data:
            parts = line.replace("\n","\t").strip().split("\t")

            #create vocabulary from all unique words in all sentences
            sentence = parts[3] + " " + parts[4]
            words = sentence.replace("'s","").lower().split()
            #add if not already in vocab
            for word in words:
                if word not in vocab:
                    #add word to vocab dict
                    vocab[word] = len(vocab)
        return vocab,len(vocab)
    


def sen2vec(s,vocab):
    v = []
    words = s.replace("'s","").lower().split()
    for word in words:
        try:
            v.append(vocab[word])
        except:
            v.append(vocab["<UNK>"])
    return tensor(v).unsqueeze(0)


def sen2glove(s,glove_embs):
    v = []
    words = s.replace("'s","").lower().split()
    for word in words:
        try:
            v.append(glove_embs.get_index(word, default=None))
        except:
            v.append(40000)
    return tensor(v).unsqueeze(0)

def sen2pos(s,pos_dict):
    v = []
    words = s.replace("'s","").lower().split()
    pos_tags = pos_tag(words)

    for word, pos in pos_tags:
        if pos in pos_dict.keys():
            embed = np.zeros((n,), dtype=np.float32)
            embed[pos_dict[pos]]=1
        else:
            # print(pos)
            embed = np.zeros((n,), dtype=np.float32)
        v.append(embed)
    return tensor(np.array(v)).unsqueeze(0)

def sentence_tag(s, pos_dict):
    v = []
    words = s.replace("'s","").lower().split()
    pos_tags = pos_tag(words)

    for word, pos in pos_tags:
        if pos in pos_dict.keys():
            v.append(pos_dict[pos])
        else:
            # print(pos)
            v.append(1000)
    return v