# coding: UTF-8
# @author: Xuecong Zheng
# time: 2023/9/23
import hp_config
import numpy as np
import Preprocess
import KING
import os
import torch
from train_eval_step import *

if __name__ == '__main__':
    np.random.seed(0)
    #torch.manual_seed(1)
    #torch.cuda.manual_seed_all(1) # 保证结果一致

    word2id, word_embeddings, train_data, train_y, test_data, length_train, length_test = Preprocess.load_data_and_labels_fewshot()
    model_config = KnowST.model_config()
    train_model = KnowST.Model(word_embeddings, model_config)
    train_iter = build_iterator(train_data, model_config)
    test_iter = Preprocess.build_iterator(test_data, model_config)

    train(model_config, train_model, train_iter, test_iter)


