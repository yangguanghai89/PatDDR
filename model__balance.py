import pdb
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class net(nn.Module):
    def __init__(self, bert_model, tokenizer,p_ipm):
        super(net,self).__init__()
        self.p_ipm = p_ipm
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.sample_weight = nn.Parameter(torch.ones(9952,1))
        self.softmax = nn.Softmax(dim = 1)

        self.phi = nn.Sequential(
            nn.Linear(768,768),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(768, 512),
            nn.Dropout(p=0.1),
            nn.Tanh()
        )

        self.map_t = nn.Sequential(
            nn.Linear(1,32),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(32, 64)
        )

        self.y0 = nn.Sequential(
            nn.Linear(576, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256,2)
        )

        self.y1 = nn.Sequential(
            nn.Linear(576, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256,2)
        )

    def forward(self, text_a, text_b, t, index, bu, dalei, tr):
        self.tr = tr
        self.bu = bu
        self.dalei = dalei
        self.t = t
        self.index = index
        text_embedding = []

        for a, b in zip(text_a, text_b):
            encode_text = self.tokenizer(a, b, add_special_tokens = True, return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 512 )
            encoded_dict = encode_text.to(device)
            outputs = self.bert_model(**encoded_dict)
            result = outputs.last_hidden_state[:, 0, :]
            text_embedding.append(result)

        self.text_embedding = torch.stack(text_embedding).squeeze(1)

        self.rep_c = self.phi(self.text_embedding)
        self.t_map = self.map_t(t.unsqueeze(1))

        y0_hat_t = self.y0(torch.cat((self.rep_c, self.t_map), dim = 1))
        y1_hat_t = self.y1(torch.cat((self.rep_c, self.t_map), dim = 1))

        y_hat_t = ((1 - t).unsqueeze(1) * y0_hat_t + t.unsqueeze(1) * y1_hat_t)
        pred_y = self.softmax(y_hat_t)[:, 1].unsqueeze(-1)

        self.i0 = [i for i, value in enumerate(t) if value == 0]
        self.i1 = [i for i, value in enumerate(t) if value == 1]

        return pred_y

    def ipm(self):
        ipm = 0
        if self.tr:
            if self.i0 or self.i1:
                c = self.sample_weight[self.index] * self.text_embedding
                mean_c_0 = c[self.i0].mean(dim = 0).unsqueeze(0)
                mean_c_1 = c[self.i1].mean(dim = 0).unsqueeze(0)
                ipm = 1 - F.cosine_similarity(mean_c_0, mean_c_1)
        else:
            if self.i0 or self.i1:
                mean_c_0 = self.text_embedding[self.i0].mean(dim = 0).unsqueeze(0)
                mean_c_1 = self.text_embedding[self.i1].mean(dim = 0).unsqueeze(0)
                ipm = 1 - F.cosine_similarity(mean_c_0, mean_c_1)

        return ipm.abs()

    def lossFunc(self, y_true, t_true, predictions):
        y = y_true
        y_pred = predictions[ : , 0]
        ipm = self.ipm()

        loss_y = self.log_loss(y_pred, y)

        return loss_y + ipm

    def log_loss(self, pred, label):
        if self.tr:
            loss_a = self.sample_weight[self.index] * F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return  loss_a.mean()

