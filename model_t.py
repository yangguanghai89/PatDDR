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

        self.t_pred = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 2)
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

        t_hat = self.t_pred(self.rep_c)
        y0_hat_t = self.y0(torch.cat((self.rep_c, self.t_map), dim = 1))
        y1_hat_t = self.y1(torch.cat((self.rep_c, self.t_map), dim = 1))

        y_hat_t = ((1 - t).unsqueeze(1) * y0_hat_t + t.unsqueeze(1) * y1_hat_t)
        pred_y = self.softmax(y_hat_t)[:, 1].unsqueeze(-1)
        pred_t = self.softmax(t_hat)[:, 1].unsqueeze(-1)

        return torch.cat((pred_y, pred_t), dim = 1)

    def lossFunc(self, y_true, t_true, predictions):
        y = y_true
        t = t_true
        y_pred = predictions[ : , 0]
        t_pred = predictions[ : , 1]
        loss_t = self.log_loss(t_pred, t)
        loss_y = self.log_loss(y_pred, y)

        return loss_t + loss_y

    def log_loss(self, pred, label):
        if self.tr:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return  loss_a.mean()

