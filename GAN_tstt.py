import torch
import os
import torch.nn as nn
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
import numpy as  np
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
# 潜在的空间 其实GAN 训练出来的判别器对对抗样例的防御是有意义的。但是
from collections import defaultdict




src_dir = r'E:\kupyter\故障诊断\故障诊断\data\sim_feature'
nor_dir = os.path.join(src_dir, 'nor_feature.csv')
unb_dir = os.path.join(src_dir, 'unb_feature.csv')
mis_dir = os.path.join(src_dir, 'mis_feature.csv')
rub_dir = os.path.join(src_dir, 'rub_feature.csv')

nor = np.array(pd.read_csv(nor_dir))
unb = np.array(pd.read_csv(unb_dir))
mis = np.array(pd.read_csv(mis_dir))
rub = np.array(pd.read_csv(rub_dir))

# 归一化
# (X - Xmin) / (Xmax - Xmin)
# 去除orbit一列
for i in range(nor.shape[1] - 1):
    nor[:, i] = (nor[:, i] - min(nor[:, i])) / (max(nor[:, i]) - min(nor[:, i]))
    unb[:, i] = (unb[:, i] - min(unb[:, i])) / (max(unb[:, i]) - min(unb[:, i]))
    mis[:, i] = (mis[:, i] - min(mis[:, i])) / (max(mis[:, i]) - min(mis[:, i]))
    rub[:, i] = (rub[:, i] - min(rub[:, i])) / (max(rub[:, i]) - min(rub[:, i]))
nor_tsne=nor
unb_tsne=unb
mis_tsne=mis
rub_tsne=rub



X=nor_tsne[:,:-1]
X=torch.tensor(X).float()


X_pca=PCA(n_components=0.90)
X_pca.fit(X)
X_pca=X_pca.transform(X)
X = torch.tensor(X_pca).float()
X = torch.tensor(datasets.load_iris().data).float()

print(X.shape)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(100, 256),

            nn.ReLU(True),

            nn.Linear(256, X.shape[1]),

        )

    def forward(self, input):
        return self.main(input)
"""#输入Z维度100, MLP 100 -256 - 4 ,4维数据input"""

# print(netG)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(

            nn.Linear(X.shape[1], 256),

            nn.ReLU(True),

            nn.Linear(256, 1),

            nn.Sigmoid()

        )

    def forward(self, input):
        return self.main(input)


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(history['g_loss'], label='g loss')
    ax1.plot(history['d_loss'], label='d loss')

    ax1.set_ylim([0, 2])
    ax1.legend()
    ax1.set_ylabel('D_G_Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(history['fake_loss'], label='fake loss')
    ax2.plot(history['real_loss'], label='real loss')

    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.set_ylabel('fake_loss')
    ax2.set_xlabel('Epoch')

    fig.suptitle('Training History')
    plt.show()


batch_size=50
num_epoch=2000
generator = Generator()
discriminator = Discriminator()


g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))#, weight_decay=0.0001

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))#, weight_decay=0.0001

loss_fn = nn.BCELoss()
labels_one = torch.ones(batch_size, 1)
labels_zero = torch.zeros(batch_size, 1)

history = defaultdict(list)  # 构建一个默认value为list的字典

data_loader = DataLoader(dataset=X, batch_size=batch_size, shuffle=True, drop_last=False)
for epoch in range(num_epoch):
    for i, mini_batch in enumerate(data_loader):







        gt_data = mini_batch
        z = torch.randn(batch_size, 100)
        pred_data = generator(z)
        g_optimizer.zero_grad()
        recons_loss = torch.abs(pred_data - gt_data).mean()
        g_loss = recons_loss * 0.05 + loss_fn(discriminator(pred_data), labels_one)
        g_loss.backward()
        g_optimizer.step()
        d_optimizer.zero_grad()
        real_loss = loss_fn(discriminator(gt_data), labels_one)
        fake_loss = loss_fn(discriminator(pred_data.detach()), labels_zero)
        d_loss = (real_loss + fake_loss)

        # 观察real_loss与fake_loss，同时下降同时达到最小值，并且差不多大，说明D已经稳定了

        d_loss.backward()
        d_optimizer.step()







        if epoch % 1 == 0:
            print(
                f"step:{len(data_loader) * epoch + i}, recons_loss:{recons_loss.item()}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
            #  netD.zero_grad()

            history['g_loss'].append(g_loss.item())
            history['d_loss'].append(d_loss.item())
            history['fake_loss'].append(fake_loss.item())
            history['real_loss'].append(real_loss.item())


#
plot_training_history(history)

"""
noise = torch.randn(X.shape[0], 100)[0]

print(netG(noise))"""
