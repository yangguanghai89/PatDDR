import pdb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class net(nn.Module):
    def __init__(self, bert_model, tokenizer, p_ipm):
        super(net,self).__init__()
        self.p_ipm = p_ipm
        self.bert_model = bert_model
        self.tokenizer = tokenizer
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

        self.t_c = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_c = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256,2)
        )

        self.y1_c = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256,2)
        )

        self.t_a = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_a = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y1_a = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.y0_a_c = nn.Sequential(
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

        self.y1_a_c = nn.Sequential(
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

        self.text = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(768, 512)
        )

        self.inv = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 256)
        )
    
        self.app = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 256)
        )


    def forward(self, state, labels, text_a, text_b, sClass, section, bClass, index, inv_flag, app_flag, tr): #text_a, text_b, sClass, index, section, bClass, inv_flag, app_flag, True
        self.tr = tr
        self.t = sClass
        self.index = index
        self.labels = labels
        text_embedding = []

        for a, b in zip(text_a, text_b):
            encode_text = self.tokenizer(a, b, add_special_tokens = True, return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 512 )
            encoded_dict = encode_text.to(device)
            outputs = self.bert_model(**encoded_dict)
            result = outputs.last_hidden_state[:, 0, :]
            text_embedding.append(result)

        self.text_embedding = torch.stack(text_embedding).squeeze(1)
        self.text_embedding = self.text(self.text_embedding)
        self.inv_embedding = self.inv(inv_flag.unsqueeze(1))
        self.app_embedding = self.app(app_flag.unsqueeze(1))

        self.feature = torch.cat((self.text_embedding, self.inv_embedding, self.app_embedding), dim=1)

        self.x_c = self.x_c_func(self.feature)
        t_c_hat = self.t_c(self.x_c)
        self.pred_t_c = self.softmax(t_c_hat)[:, 1].unsqueeze(-1)

        y0_hat_c = self.y0_c(self.x_c)
        y1_hat_c = self.y1_c(self.x_c)
        y_hat_c = ((1 - self.t).unsqueeze(1) * y0_hat_c + self.t.unsqueeze(1) * y1_hat_c)
        self.pred_y_c = self.softmax(y_hat_c)[:, 1].unsqueeze(-1)

        self.x_a = self.x_a_func(self.feature)
        t_a_hat = self.t_a(self.x_a)
        self.pred_t_a = self.softmax(t_a_hat)[:, 1].unsqueeze(-1)

        y0_hat_a = self.y0_a(self.x_a)
        y1_hat_a = self.y1_a(self.x_a)
        y_hat_a = ((1 - self.t).unsqueeze(1) * y0_hat_a + self.t.unsqueeze(1) * y1_hat_a)
        self.pred_y_a = self.softmax(y_hat_a)[:, 1].unsqueeze(-1)

        x_a_c = torch.cat((self.x_c, self.x_a), dim=1)
        y0_hat_a_c = self.y0_a_c(x_a_c)
        y1_hat_a_c = self.y1_a_c(x_a_c)
        y_hat_a_c = ((1 - self.t).unsqueeze(1) * y0_hat_a_c + self.t.unsqueeze(1) * y1_hat_a_c)
        self.pred_y_a_c = self.softmax(y_hat_a_c)[:, 1].unsqueeze(-1)

        self.i0 = [i for i, value in enumerate(self.t) if value == 0]
        self.i1 = [i for i, value in enumerate(self.t) if value == 1]

        self.pred_y = self.pred_y_a_c


    def ipm(self, feature, bw=False):
        ipm = 0
        if bw:
            if self.i0 and self.i1:
                # c = self.sample_weight[self.index] * self.text_embedding
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

    def lossFunc(self):  #, pred_t_c, pred_y_c, pred_t_a, pred_y_a, pred_y_a_c):

        y_true = self.labels

        loss_a = self.log_loss(self.pred_y_a, y_true, False) + self.log_loss(self.pred_t_a, 1 - self.t, False) #+ self.ipm(self.x_a, bw=False)

        #loss_c = self.log_loss(self.pred_y_c, y_true, False) + self.log_loss(self.pred_t_c, self.t, False) + self.ipm(self.x_c, bw=True)
        loss_c = self.log_loss(self.pred_y_c, y_true, False) + self.log_loss(self.pred_t_c, self.t, False) #+ self.ipm(self.x_c, bw=False)

        #loss_y_a_c = self.log_loss(self.pred_y_a_c, y_true, True)
        loss_y_a_c = self.log_loss(self.pred_y_a_c, y_true, False)

        return loss_a + loss_c + loss_y_a_c
        #return loss_a

    def log_loss(self, pred, label, bw=False):
        pred = pred.squeeze(dim=1)
        if bw:
            loss_a = self.sample_weight[self.index] * F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return loss_a.mean()



