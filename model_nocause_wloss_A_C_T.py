
import torch
from torch import nn
import torch.nn.functional as F
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class net(nn.Module):
    def __init__(self, bert_model, tokenizer, p_ipm):
        super(net,self).__init__()
        self.p_ipm = p_ipm
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.sample_weight = nn.Parameter(torch.ones(9952,1))
        self.softmax = nn.Softmax(dim = 1)
        drop=0.05

        self.text = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(768, 512)
        )

        self.inv = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(256, 256)
        )
    
        self.app = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(256, 256)
        )

        self.ipc = nn.Sequential(
            nn.Linear(1, 256),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(256, 256)
        )

        self.y_a_c_i_func = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Dropout(p=drop),
            nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, text_a, text_b, t, index, bu, dalei, inv_flag, app_flag, tr):
        self.tr = tr
        self.bu = bu
        self.dalei = dalei
        self.t = t
        self.index = index
        text_embedding = []

        for a, b in zip(text_a, text_b):
            encode_text = self.tokenizer(a, b, add_special_tokens = True, return_tensors = 'pt', padding = 'max_length', truncation = True, max_length = 128 )
            encoded_dict = encode_text.to(device)
            outputs = self.bert_model(**encoded_dict)
            result = outputs.last_hidden_state[:, 0, :]
            text_embedding.append(result)

        self.text_embedding = torch.stack(text_embedding).squeeze(1)
        self.text_embedding = self.text(self.text_embedding)
        self.inv_embedding = self.inv(inv_flag.unsqueeze(1))
        self.app_embedding = self.app(app_flag.unsqueeze(1))
        self.ipc_embedding = self.ipc(t.unsqueeze(1))

        self.feature = self.text_embedding
        #self.feature = self.transform(self.feature)

        pred_y = self.y_a_c_i_func(self.feature)
        pred_y = self.softmax(pred_y)[:, 1].unsqueeze(-1)
        #pred_y = self.softmax(pred_y)

        return pred_y

    def ipm(self, feature):
        ipm = 0
        if self.tr:
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

    def lossFunc(self, y_true, pred_y):
        '''
        y_0 = torch.zeros(pred_y.shape[0], pred_y.shape[1]).to(device)
        y_0[range(y_0.shape[0]), y_true.int()] = 1.0
        y_true = y_0
        loss = self.log_loss(pred_y, y_true, False)
        '''

        return loss

    def log_loss(self, pred, label, tr):

        pred = pred.squeeze(dim=1)
        if tr:
            loss_a = self.sample_weight[self.index] * F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            #loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return loss_a.mean()

    def get_w_loss(self):
        loss = 0.0
        if self.i0 and self.i1:
            index_i0 = self.index[self.i0]
            index_i1 = self.index[self.i1]
            count_i0 = torch.sum(1.0 - self.t)
            count_i1 = torch.sum(self.t)
            avg_i0 = torch.sum(self.sample_weight[index_i0]) / count_i0
            avg_i1 = torch.sum(self.sample_weight[index_i1]) / count_i1
            loss = torch.square(avg_i0-1.0) + torch.square(torch.sum(avg_i1)-1.0)
        return loss



