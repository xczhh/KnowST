import numpy as np
import os
import time
import datetime
import random
import hp_config
import torch
import torch.nn.functional as F
from transformers import logging, AutoTokenizer, AutoModel
from gensim.models import word2vec
#from transformers import logging, AutoTokenizer,BertTokenizerFast
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def load_data_and_labels_fewshot():
    path = os.getcwd()
    data_path = data_path 
    vec_path = pretrain_path 
    vec_path = vec_path + "pretrain-300D"
    train_path = data_path + "train"
    test_path = data_path + "test"

    # step1: init word embedding
    word_embeddings = []
    word2id = {}
    f = open(vec_path, "r", encoding='utf-8')
    content = f.readline()
    while True:
        content = f.readline()
        if content == "":
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)  # word->id
        content = content[1:]
        content = [(float)(i) for i in content]
        word_embeddings.append(content)

    f.close()
    word2id['UNK'] = len(word2id)  
    word2id['BLANK'] = len(word2id)
    lists = [0.0 for i in range(len(word_embeddings[0]))]  # lists: [100]
    word_embeddings.append(lists)
    word_embeddings.append(lists)
 

    # step2 : init train set and train_attrs
    train_x = []
    train_y = []
    train_end_attrs = []
    train_early_attrs1 = []
    train_early_attrs2 = []
    train_data= []

    f = open(train_path, "r", encoding='utf-8')
    content = f.readlines()  # content[6]
    random.shuffle(content)
    for i in content:
        z = i.strip().split("\t")
        if(len(z) != 4): continue

        temp = []
        for j in z[1].strip().split():
            temp.append(int(j))

        y_end_attr = []
        y_early_attr1 = []
        y_early_attr2 = []
        for j in z[2].strip().split():
            y_end_attr.append(int(j))
        p = z[3].strip().split()
        y_early_attr1.append(int(p[0]))
        y_early_attr2.append(int(p[1]))
        if len(y_end_attr) != hp_config.num_of_end:
            print("end attributes error")
            continue

        train_x.append(z[0])
        train_y.append(temp)
        train_y1= train_y
        train_end_attrs.append(y_end_attr)
        train_early_attrs1.append(y_early_attr1)
        train_early_attrs2.append(y_early_attr2)
    f.close()
    if len(train_x)==len(aug_similiar_x): print('equal')
    print(len(train_x))
    print(len(train_y))

    # step3: init test_oringinal set
    test_x = []
    test_y = []
    test_similiar_x = []
    test_end_attrs = []
    test_early_attrs1 = []
    test_early_attrs2 = []
    test_data = []
    f = open(test_path, "r", encoding='utf-8')
    content = f.readlines()
    for i in content:
        z = i.strip().split("\t")
        if (len(z) != 4): continue

        temp = []
        for j in z[1].strip().split():
            temp.append(int(j))

        y_end_attr = []
        y_early_attr1 = []
        y_early_attr2 = []
        for j in z[2].strip().split():
            y_end_attr.append(int(j))
        p = z[3].strip().split()
        y_early_attr1.append(int(p[0]))
        y_early_attr2.append(int(p[1]))
        if len(y_end_attr) != hp_config.num_of_end:
            print("end attributes error")
            continue

        test_x.append(z[0])
        test_y.append(temp)
        test_y1 = test_y
        test_end_attrs.append(y_end_attr)
        test_early_attrs1.append(y_early_attr1)
        test_early_attrs2.append(y_early_attr2)
    f.close()
    print("test_x: " + str(len(test_x)))
    print("test_y" + str(len(test_y)))


    # step4: class categorization
    res = []
    for i in range(0, len(test_y)):
        label = [0 for k in range(0, hp_config.classes_of_st)]
        for j in test_y[i]:
            label[j] = 1
        res.append(label)
    test_y = np.array(res)

    res = []
    for i in range(0, len(train_y)):
        label = [0 for k in range(0, hp_config.classes_of_st)]
        for j in train_y[i]:
            label[j] = 1
        res.append(label)
    train_y = np.array(res)

    # step5:

    #tokenizer = AutoTokenizer.from_pretrained(hp_config.pretrain_path)

    max_sequence = 200
    size = len(train_x)
    length_train = []
    length_test = []
    blank = word2id['BLANK']
    # text = tokenizer(train_x,max_length=200, padding=True, truncation=True, return_tensors="pt")
    # train_x = np.array(text["input_ids"])
    for i in range(size):
        length_train.append(200)
    for i in range(size):
        text = [blank for j in range(max_sequence)]
        content = train_x[i].split()
        for j in range(len(content)):
            if (j == max_sequence):
                break
            if not content[j] in word2id:
                text[j] = word2id['UNK']
            else:
                text[j] = word2id[content[j]]
        train_x[i] = text


    train_x = np.array(train_x)
    

    length_train = np.array(length_train)

    size = len(test_x)
    
    for i in range(size):
        length_test.append(200)
    for i in range(size):
        text =[blank for k in range(max_sequence)]
        content = test_x[i].split()
        for j in range(len(content)):
            if (j == max_sequence):
                break
            if not content[j] in word2id:
                text[j] = word2id['UNK']
            else:
                text[j] = word2id[content[j]]
        length_test.append(min(max_sequence,len(content)))
        test_x[i] = text
   
    test_x = np.array(test_x)
    
    length_test = np.array(length_test)


    for i in range(0, len(train_x)):
        train_data.append((train_x[i],  train_y[i], train_end_attrs[i], train_early_attrs1[i], train_early_attrs2[i], length_train[i]))

    for i in range(0, len(test_x)):
        test_data.append((test_x[i],  test_y[i], test_end_attrs[i], test_early_attrs1[i], test_early_attrs2[i], length_test[i]))


    return word2id, word_embeddings, train_data, train_y, test_data, length_train, length_test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, num_end_attrs, num_early_attrs1, num_early_attrs2):
        self.batch_size = batch_size
        self.num_end_attrs = num_end_attrs
        self.num_early_attrs1 = num_early_attrs1
        self.num_early_attrs2 = num_early_attrs2
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas, num_end_attrs, num_early_attrs1, num_early_attrs2):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        
        attrs_end = F.one_hot(torch.LongTensor([_[2] for _ in datas]), num_classes=num_end_attrs).to(self.device)

        ea1 = F.one_hot(torch.LongTensor([line[3] for line in datas]), num_classes=3)
        ea2 = F.one_hot(torch.LongTensor([line[4] for line in datas]), num_classes=4)

       
        seq_len = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        return (x, seq_len), y, attrs_end, ea1, ea2

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches, self.num_end_attrs, self.num_early_attrs1, self.num_early_attrs2)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches, self.num_end_attrs, self.num_early_attrs1, self.num_early_attrs2)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.num_end_attrs, config.num_early_attrs1, config.num_early_attrs2)
    return iter


if __name__ == '__main__':
    load_data_and_labels_fewshot()
