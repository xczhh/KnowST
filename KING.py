# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import hp_config
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.autograd import Variable
from transformers import logging, AutoTokenizer, AutoModel

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class model_config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'TextRNN_Att'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.embed = 300  # dim
        self.dropout = 0.7  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.cnn_dropout = 0.5

        self.max_sequence = 300
        self.vocab_size = 10000  # 词表大小，在运行时赋值
        self.num_epochs = 50  # epoch数
        self.batch_size = 32  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3  # 学习率

        self.num_classes = 6  # 类别数
        self.num_end_attrs = 4
        self.num_early_attrs1 = 3
        self.num_early_attrs2 = 4

        self.hidden_loss1 = 7

        self.hidden_size = 150  # lstm隐藏层 128 transformer 150
        self.num_layers = 2  # lstm层数
        self.hidden_size2 = 64
        self.ave_ratio = 10
        self.ratio = 1.2
        self.test_bs = 56
        path = os.getcwd()
        self.save_path = path + '/saved_dict/' + self.model_name


        # transformer config list
        self.word_limit = 200
        self.min_word_count = 5
        self.fine_tune_word_embeddings = True
        self.t_hidden_size = 512
        self.n_heads = 6
        self.n_encoders = 2
        self.t_dropout = 0.3


'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss



class GRUDownsample(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUDownsample, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 将输入形状转换为 [batch_size, sequence_length, input_dim]
        x = x.permute(0, 2, 1)

        # 将输入张量传递给 GRU 模型
        _, h = self.gru(x)

        # 取最后一个时间步的隐藏状态作为输出
        output = self.fc(h.squeeze(0))

        return output


class Model(nn.Module):
    def __init__(self, word_embeddings, config):
        super(Model, self).__init__()

        self.ave_ratio = config.ave_ratio
        self.batch = config.batch_size
        self.hidden = config.hidden_size * 2
        self.hidden_size = config.hidden_size
        self.num_end_attrs = config.num_end_attrs
        self.num_early_attrs1 = config.num_early_attrs1
        self.num_early_attrs2 = config.num_early_attrs2
        self.kernel_size = [2, 3, 4]
        self.device = config.device
        
        self.temperature = 0.07
        self.temp = 0.07

        self.embedding = nn.Embedding.from_pretrained(torch.tensor(word_embeddings), freeze=False)
        self.convs = nn.ModuleList([nn.Conv2d(1, 100, (K, 300)) for K in self.kernel_size])
        self.cnn_dropout = nn.Dropout(config.cnn_dropout)
        self.out_convs = nn.Conv1d(1, 6, 2)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        
        self.GRU = GRUDownsample(input_dim=200, hidden_dim=100, output_dim=300)
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc = nn.Linear(config.hidden_size * 2, 100)
        self.ln1 = nn.LayerNorm([150, 300], eps=1e-05, elementwise_affine=True, device=None, dtype=None)
        self.ln2 = nn.LayerNorm([64, 300], eps=1e-05, elementwise_affine=True, device=None, dtype=None)
        self.fc_stage_1 = nn.Linear(400, 200)  # 400
        self.fc_stage = nn.Linear(200, config.num_classes)
        self.fc_end_attrs = nn.Linear(100, config.num_end_attrs)
        self.fc_early_attrs1 = nn.Linear(100, 3)
        self.fc_early_attrs2 = nn.Linear(100, 4)
        
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.x=0.5
        self.y=0.1


    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = torch.max(labels, 1)[1]
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1)) 
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        # positives
        logits = logits * mask.cpu()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        pos_logits = (mask.cpu() * log_prob).sum(dim=1) / mask_sum.detach().cpu()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, x, target, attrs_end, attrs_ea1, attrs_ea2, train=False):
        if train:

            features = x[0]

            attrs_early1 = attrs_ea1
            attrs_early2 = attrs_ea2
          
            similiar_emb = self.embedding(features)
            
            similiar_trans = self.GRU(similiar_emb)

            cl_loss = self.nt_xent_loss(similiar_trans, similiar_trans, target)

            output_CNN = similiar_emb.unsqueeze(1)

            output_CNN1 = [F.relu(conv(output_CNN)).squeeze(3) for conv in self.convs]
            output_CNN2 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output_CNN1]
            output_CNN3 = torch.cat(output_CNN2, 1)
            output_CNN4 = self.cnn_dropout(output_CNN3)
            out_fc = self.fc(output_CNN4)
            attr_end_pred = self.fc_end_attrs(out_fc)#classifier_1(similiar_trans)


            loss2 = 0.0
            loss2_end_output = 0.0
            i=0
            j=0
            loss2_early_output = 0.0
            attr_end_ind = torch.max(attrs_end.squeeze(1), 1)[1]
            for i in range(0, len(attr_end_pred)):
                loss2_end = F.cross_entropy(attr_end_pred[i].unsqueeze(0), attr_end_ind[i].unsqueeze(0))
                attr_early_pred1 = self.fc_early_attrs1(out_fc[i])
                attr_early_pred2 = self.fc_early_attrs2(out_fc[i])
                
                loss2_early1 = F.cross_entropy(attr_early_pred1.unsqueeze(0), attrs_early1[i].cuda().float())
                loss2_early2 = F.cross_entropy(attr_early_pred2.unsqueeze(0), attrs_early2[i].cuda().float())
                p = torch.max(attr_end_pred[i].unsqueeze(0), 1)[1]
                p = p.cpu().numpy()
                if p[0] > 2:
                    i+=1
                    loss2_early_output += loss2_early1 + loss2_early2
                    loss2 += loss2_early1 + loss2_early2
                else:
                    j+=1
                    loss2_end_output += loss2_end
                    loss2 += loss2_end
            loss2 /= features.size()[0]
            if i != 0:
                loss2_early_output /= i
            if j != 0:
                loss2_end_output /= j

            loss2 = self.x * loss2 + self.y * cl_loss

            temp = [similiar_trans, out_fc]
            fusion_output = torch.cat(temp, dim=1)
            stage_p1 = self.fc_stage_1(fusion_output)
            stage_pred = self.fc_stage(stage_p1)
           
            return stage_pred, loss2, loss2_end_output, loss2_early_output


        else:
            x, words_per_sentence = x
            emb = self.embedding(x.cpu())  # [batch_size, seq_len, embeding]=[128, 32, 300]
            outputs_trans = self.GRU(emb)

            attr_early1 = []
            attr_early2 = []

            output_CNN = emb.unsqueeze(1)
            output_CNN = [F.relu(conv(output_CNN)).squeeze(3) for conv in self.convs]
            output_CNN = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output_CNN]
            output_CNN = torch.cat(output_CNN, 1)
            output_CNN = self.cnn_dropout(output_CNN)
            out_fc = self.fc(output_CNN)
            attr_end_pred = self.fc_end_attrs(out_fc)

            loss2 = 0.0
            i=0.0
            j=0.0
            loss2_early_output = 0.0
            loss2_end_output = 0.0
            attr_end_ind = torch.max(attrs_end.squeeze(1), 1)[1]
            for i in range(0, len(attr_end_pred)):
                loss2_end = F.cross_entropy(attr_end_pred[i].unsqueeze(0).cuda(), attr_end_ind[i].unsqueeze(0))
                attr_early_pred1 = self.fc_early_attrs1(out_fc[i])
                attr_early_pred2 = self.fc_early_attrs2(out_fc[i])
                loss2_early1 = F.cross_entropy(attr_early_pred1.unsqueeze(0).cuda(), attrs_ea1[i].cuda().float())
                loss2_early2 = F.cross_entropy(attr_early_pred2.unsqueeze(0).cuda(), attrs_ea2[i].cuda().float())
                attr_early1.append(attr_early_pred1)
                attr_early2.append(attr_early_pred2)
                p = torch.max(attr_end_pred[i].unsqueeze(0), 1)[1]
                p = p.cpu().numpy()
                if p[0] > 2:
                    i += 1
                    loss2_early_output +=loss2_early1 + loss2_early2
                    loss2 += loss2_early1 + loss2_early2
                else:
                    j += 1
                    loss2_end_output += loss2_end
                    loss2 += loss2_end
            loss2 /= emb.size()[0]
            if i!=0:
                loss2_early_output /= i
            if j!=0:
                loss2_end_output /= j

            cl_loss = self.nt_xent_loss(outputs_trans, outputs_trans, target)
            loss2= self.x * loss2 + self.y * cl_loss
            

            temp = [outputs_trans, out_fc]
            fusion_output = torch.cat(temp, dim=1)
            stage_p1=self.fc_stage_1(fusion_output)
            stage_pred = self.fc_stage(stage_p1)
           

            return stage_pred, loss2, 0, stage_p1, attr_end_pred, attr_early1, attr_early2, loss2_end_output, loss2_early_output


if __name__ == '__main__':
    data = np.random.randint(0, 100, size=(8, 32, 300))
    print(data)
