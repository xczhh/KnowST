# coding: UTF-8
# author: Xc Zheng
# time : 2023/9/24

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from Preprocess import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix

train_loss_ = []
loss_task1 = []
loss_task2 = []
loss_task3 = []
loss_task4 = []
test_loss1 = []
test_loss2 = []
test_loss_st =[]
torch.manual_seed(0)
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


test_best_loss = float('inf')

def sen(Y_test, Y_pred, n):  # n为分类数

    sen = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen


def spe(Y_test, Y_pred, n):
    spe = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def train(config, model, train_iter,  test_iter):
    print("train start!")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = FocalLoss(gamma=2, weight=None)

    total_batch = 0
    torch.backends.cudnn.enabled = False
    for epoch in range(config.num_epochs):
        val_loss, val_acc = 0.0, 0.0
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains,  labels, attrs_end, attrs_ea1, attrs_ea2) in enumerate(train_iter):
            model = model.to(config.device)
            outputs, loss2, loss2_end_output, loss2_early_output = model(trains, labels,  attrs_end, attrs_ea1, attrs_ea2, train=True)
            
            model.zero_grad()
            labels = labels.float()
            
            attrs_end = attrs_end.float()
            loss1 = criterion(outputs, labels)
            loss_total = loss1 + loss2  
            train_loss_.append(loss_total)
            loss_task1.append(loss1)
            loss_task2.append(loss2)
            loss_task3.append(loss2_end_output)
            loss_task4.append(loss2_early_output)

            loss_total.backward()
            optimizer.step()
            label = labels.cpu()  
            label = torch.max(label, 1)[1]
            pred = torch.max(outputs.data, 1)[1].cpu()
            acc = metrics.accuracy_score(label, pred)
            msg = 'Iter: {0:>6}, Loss: {1:>5.2}, Acc: {2: >6.2%}'
            print(msg.format(total_batch, loss_total.item(), acc))
            total_batch += 1
            model.train()

        if epoch % 1 == 0:
           print('train_epoch{},loss{:.4f},acc{:.4f}'.format(epoch, val_loss, val_acc))

        
        test(config, model, test_iter)

        torch.save(model.state_dict(), config.save_path + "_epoch_" + str(epoch) + '.ckpt')


   



def test(config, model, test_iter):
    
    model.eval()

    test_acc, test_loss, test_report, test_confusion, auc_value, f1_value, p, r = evaluate(config, model, test_iter, test=True)

    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print("auc_value:", auc_value)
    print("f1_value:", f1_value)
    print("p:", p)
    print("r:", r)




def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    auc_pred = []
    auc_label = []
    pred_total = np.array([], dtype=int)
    label_total = np.array([], dtype=int)
    criterion = FocalLoss(gamma=2, weight=None)
    with torch.no_grad():
        for texts, labels, attrs_end, attrs_ea1, attrs_ea2 in data_iter:
            outputs, loss2, loss_cl,fusion_output, attr_end_pred, attr_early_pred1, attr_early_pred2, loss2_end_output, loss2_early_output = model(texts,labels, attrs_end, attrs_ea1, attrs_ea2, train=False)
            labels = labels.float()
            loss = criterion(outputs, labels)
            test_loss2.append(loss2/outputs.size()[0])
            test_loss_st.append(loss/outputs.size()[0])
            loss_total += loss + loss2
            test_loss1.append(loss_total/outputs.size()[0])
            labels = labels.cpu()
            auc_label.append(labels)
            auc_pred.append(outputs.data)
            labels = torch.max(labels, 1)[1].numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            label_total = np.append(label_total, labels)
            pred_total = np.append(pred_total, predic)

    acc = metrics.accuracy_score(label_total, pred_total)
    auc_pred1 = torch.cat(auc_pred, dim=0)
    auc_label1 = torch.cat(auc_label, dim=0)
    auc_value = roc_auc_score(auc_label1.cpu(), auc_pred1.cpu(), multi_class='ovo', average='macro')
    f1_value = f1_score(label_total, pred_total, average='macro')
    p = precision_score(label_total, pred_total, average='macro')
    r = recall_score(label_total, pred_total, average='macro')
    print("sensitive ->", sen(label_total, pred_total, 6))
    print("specificity ->", spe(label_total, pred_total, 6))
    if test:
        class_list = ['1a', '1b', '2a', '2b', '3a', '3b']
        report = metrics.classification_report(label_total, pred_total, target_names=class_list,digits=4)
        confusion = metrics.confusion_matrix(label_total, pred_total)
        print("------pred_total-------")
        print(pred_total)
        print("----------end----------")
        return acc, loss_total / len(data_iter), report, confusion, auc_value, f1_value, p, r
    return acc, loss_total / len(data_iter)
