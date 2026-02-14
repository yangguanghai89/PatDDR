import pdb
import torch, random
import numpy as np
from pyarrow import nulls
from torch import nn
import torch.nn.functional as F
from P_BERT.models import Transformer, Config
from P_BERT import train, tokenization
from P_BERT.classify_pat_src import Tokenizing_with_position_count_for_prefixSPE, AddSpecialTokensWithTruncation2, TokenIndexing2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class net(nn.Module):
    def __init__(self, bert_model, tokenizer, p_ipm):
        super(net,self).__init__()
        self.p_ipm = p_ipm
        self.bert_model = bert_model

        model_cfg = 'P_BERT/config/bert_base.json'
        cfg = Config.from_json(model_cfg)
        self.transformer = Transformer(cfg)

        #
        max_len = 128
        vocab = '/home/wangfei/study/dataset/wangfei/model/uncased_L-12_H-768_A-12/vocab.txt'
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
        self.pipeline = [Tokenizing_with_position_count_for_prefixSPE(tokenizer),
                    AddSpecialTokensWithTruncation2(max_len),
                    TokenIndexing2(tokenizer.convert_tokens_to_ids,
                                   ['0', '1'], max_len)]

        self.train_tensors = None
        self.eval_tensors = None
        self.test_tensors = None

        #self.tokenizer = tokenizer


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
        self.inv_flag = inv_flag
        self.app_flag = app_flag
        self.labels = labels

        features = self.handle_data(state, text_a, text_b)
        input_ids, segment_ids, soft_position, token_count, input_mask = features[0], features[1], features[2], features[3], features[4]
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        soft_position = soft_position.to(device)
        token_count = token_count.to(device)
        input_mask = input_mask.to(device)

        outputs = self.transformer(state, input_ids, segment_ids, soft_position, token_count, input_mask)
        self.text_embedding = outputs[:, 0]

        #result = outputs.last_hidden_state[:, 0, :]
        #text_embedding.append(result)

        #self.text_embedding = torch.stack(text_embedding).squeeze(1)
        self.text_embedding = self.text(self.text_embedding)
        self.inv_embedding = self.inv(self.inv_flag.unsqueeze(1))
        self.app_embedding = self.app(self.app_flag.unsqueeze(1))

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

        loss_a = self.log_loss(self.pred_y_a, self.labels, False) + self.log_loss(self.pred_t_a, 1 - self.t, False) #+ self.ipm(self.x_a, bw=False)

        #loss_c = self.log_loss(self.pred_y_c, y_true, False) + self.log_loss(self.pred_t_c, self.t, False) + self.ipm(self.x_c, bw=True)
        loss_c = self.log_loss(self.pred_y_c, self.labels, False) + self.log_loss(self.pred_t_c, self.t, False) #+ self.ipm(self.x_c, bw=False)

        #loss_y_a_c = self.log_loss(self.pred_y_a_c, y_true, True)
        loss_y_a_c = self.log_loss(self.pred_y_a_c, self.labels, False)

        return loss_a + loss_c + loss_y_a_c
        #return loss_a

    def log_loss(self, pred, label, bw=False):
        pred = pred.squeeze(dim=1)
        if bw:
            loss_a = self.sample_weight[self.index] * F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        else:
            loss_a = F.binary_cross_entropy(input=pred, target=label.float(), reduction='none')
        return loss_a.mean()

    def handle_data(self, state, text_a, text_b):
        if state == 1:
            return self.handle_trainDataset(text_a, text_b)
        elif state == 2:
            return self.handle_evalDataset(text_a, text_b)
        else:
            return self.handle_testDataset(text_a, text_b)

    def handle_trainDataset(self, text_a, text_b):
        data = []
        for ta, tb in zip(text_a, text_b):
            # 数据增强
            instances = self.dataAugmentation2(('0', ta, tb), ratio=1)
            for instance in instances:
                for proc in self.pipeline:  # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        #print('Train example:' + str(len(data)))

        # To Tensors
        return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def handle_evalDataset(self, text_a, text_b):
        data = []
        for ta, tb in zip(text_a, text_b):
            instance = ('0', ta, tb)
            for proc in self.pipeline:
                instance = proc(instance)
            data.append(instance)

        # To Tensors
        return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def handle_testDataset(self, text_a, text_b):
        data = []
        for ta, tb in zip(text_a, text_b):
            instance = ('0', ta, tb)
            for proc in self.pipeline:
                instance = proc(instance)
            data.append(instance)

        # To Tensors
        return [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def dataAugmentation2(self, instance, ratio=2):
        instances = list()
        label, text_a, text_b = instance
        text_a = text_a.split(' ')
        text_b = text_b.split(' ')
        #val = min(len(text_a), len(text_b))
        for i in range(ratio):
            if i == 0:
                instances.append(instance)
            else:
                val = len(text_a)
                bndy = random.randint(2, val-1)
                text_a = text_a[bndy:len(text_a)] +text_a[0:bndy]
                val = len(text_b)
                bndy = random.randint(2, val - 1)
                text_b = text_b[bndy:len(text_b)] + text_b[0:bndy]
                instances.append((label, ' '.join(text_a), ' '.join(text_b)))
        return instances
