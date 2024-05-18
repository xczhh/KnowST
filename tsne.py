from sklearn.manifold import TSNE
import torch
import Preprocess
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import KING
import os
import numpy as np

# digits = load_digits()
word2id, word_embeddings, train_data, train_y, test_data, length_train, length_test = Preprocess.load_data_and_labels_fewshot()
model_config = KnowST.model_config()
train_model = KnowST.Model(word_embeddings, model_config)

train_model.load_state_dict(torch.load('saved_dict/0.9191.ckpt'))
train_model.eval()

test_iter = Preprocess.build_iterator(test_data, model_config)

targets = []
datas=[]
with torch.no_grad():
    for texts, labels, attrs_end, attrs_ea1, attrs_ea2 in test_iter:
               outputs, loss2, loss_cl,fusion_output, attr_end_pred, attr_early_pred1, attr_early_pred2, loss2_end_output, loss2_early_output = train_model(texts, labels, attrs_end, attrs_ea1, attrs_ea2, train=False)
               target= torch.max(labels, 1)[1]
               targets = torch.cat([torch.tensor(targets).cuda(), target], dim=0)
               datas = torch.cat([torch.tensor(datas), outputs], dim=0)


X_tsne = TSNE(n_components=2,random_state=33).fit_transform(datas)
X_pca = PCA(n_components=2).fit_transform(datas)

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

plt.figure(figsize=(10, 5))
plt.subplot(121)
lines = np.zeros(10) + 3

labels = ['Ia', 'Ib', 'IIa', 'IIb', 'IIIa', 'IIIb']

y1 = targets.detach().cpu().numpy()
y1.astype('int16')


for i in range(6):
    plt.scatter(X_tsne[y1 == i, 0], X_tsne[y1 == i, 1], s=20, alpha=0.8, label=labels[i])
plt.legend()
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.array(targets.cpu()),label="PCA")
plt.legend()
plt.savefig('images/0.9191.svg',format='svg', dpi=120)
plt.show()
