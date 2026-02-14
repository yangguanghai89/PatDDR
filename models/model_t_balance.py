import pdb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class model_t_balance(nn.Module):
    def __init__(self, p_ipm):
        super(model_t_balance,self).__init__()
        self.p_ipm = p_ipm
        self.sample_weight = nn.Parameter(torch.ones(42028,1))
        self.softmax = nn.Softmax(dim = 1)

        self.x_c_func = nn.Sequential(
            nn.Linear(1024,1024),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.Sigmoid()
        )

        self.x_a_func = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.Sigmoid()
        )

        self.t_c_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_c_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256,2)
        )

        self.y1_c_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256,2)
        )

        self.t_a_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_a_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y1_a_func = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_a_c_func = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y1_a_c_func = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.text_func = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(768, 512)
        )

        self.meta1_func = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 256)
        )
    
        self.meta2_func = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 256)
        )


    def forward(self, text_embedding, meta1_flag, meta2_flag, t, labels, index, tr):
        self.tr = tr
        self.t = t
        self.labels = labels
        self.index = index

        text_embedding = self.text_funct(text_embedding)
        meta1_embedding = self.meta1_func(meta1_flag)
        meta2_embedding = self.meta2_func(meta2_flag)

        self.feature = torch.cat((text_embedding, meta1_embedding, meta2_embedding), dim=1)

        x_c = self.x_c_func(self.feature)
        t_c_hat = self.t_c_func(x_c)
        self.pred_t_c = self.softmax(t_c_hat)[:, 1].unsqueeze(-1)

        y0_hat_c = self.y0_c_func(x_c)
        y1_hat_c = self.y1_c_func(x_c)
        y_hat_c = ((1 - self.t).unsqueeze(1) * y0_hat_c + self.t.unsqueeze(1) * y1_hat_c)
        self.pred_y_c = self.softmax(y_hat_c)[:, 1].unsqueeze(-1)

        x_a = self.x_a_func(self.feature)
        t_a_hat = self.t_a_func(x_a)
        self.pred_t_a = self.softmax(t_a_hat)[:, 1].unsqueeze(-1)

        y0_hat_a = self.y0_a(x_a)
        y1_hat_a = self.y1_a(x_a)
        y_hat_a = ((1 - self.t).unsqueeze(1) * y0_hat_a + self.t.unsqueeze(1) * y1_hat_a)
        self.pred_y_a = self.softmax(y_hat_a)[:, 1].unsqueeze(-1)

        x_a_c = torch.cat((x_c, x_a), dim=1)
        y0_hat_a_c = self.y0_a_c_func(x_a_c)
        y1_hat_a_c = self.y1_a_c_func(x_a_c)
        y_hat_a_c = ((1 - self.t).unsqueeze(1) * y0_hat_a_c + self.t.unsqueeze(1) * y1_hat_a_c)
        self.pred_y_a_c = self.softmax(y_hat_a_c)[:, 1].unsqueeze(-1)

        self.i0 = [i for i, value in enumerate(self.t) if value == 0]
        self.i1 = [i for i, value in enumerate(self.t) if value == 1]

        self.pred_y = self.pred_y_a_c


    def ipm(self, feature, bw=False):
        ipm = 0
        if bw:
            if self.i0 and self.i1:
                c = self.sample_weight[self.index] * feature
                mean_c_0 = c[self.i0].mean(dim=0).unsqueeze(0)
                mean_c_1 = c[self.i1].mean(dim=0).unsqueeze(0)
                ipm = (1 - F.cosine_similarity(mean_c_0, mean_c_1)).abs()
        else:
            if self.i0 and self.i1:
                mean_c_0 = feature[self.i0].mean(dim=0).unsqueeze(0)
                mean_c_1 = feature[self.i1].mean(dim=0).unsqueeze(0)
                ipm = (1 - F.cosine_similarity(mean_c_0, mean_c_1)).abs()

        return ipm

    def lossFunc(self):

        loss_a = self.log_loss(self.pred_y_a, self.labels, False) + self.log_loss(self.pred_t_a, 1 - self.t, False) + self.ipm(self.x_a, bw=False)

        loss_c = self.log_loss(self.pred_y_c, self.labels, False) + self.log_loss(self.pred_t_c, self.t, False) + self.ipm(self.x_c, bw=True)

        loss_y_a_c = self.log_loss(self.pred_y_a_c, self.labels, True)

        return loss_a + loss_c + loss_y_a_c

    def log_loss(self, pred, label, bw=False):
        pred = pred.squeeze(dim=1)
        if bw:
            loss_a = self.sample_weight[self.index] * F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return loss_a.mean()



