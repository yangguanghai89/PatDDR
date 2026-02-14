import pdb
from pyexpat import features

import torch, random
import numpy as np
from pyarrow import nulls
from torch import nn
import torch.nn.functional as F
from prefix_sequence_compression import handle_trainDataset, handle_evalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PatDDR(nn.Module):
    def __init__(self, bert_model, pipelines):
        super(PatDDR,self).__init__()

        self.softmax = nn.Softmax(dim=1)

        self.bert_model = bert_model

        self.pipelines = pipelines

        self.y_text = nn.Sequential(
            nn.Linear(768, 768),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(768, 256),
            nn.Dropout(p=0.1),
            nn.Tanh(),
            nn.Linear(256, 2)
        )

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, state, data):

        text_a = list(data[0])
        text_b = list(data[1])

        if state == 1:
            labels = data[4].to(device)
            aug_matrix = torch.tensor([[1], [1]]).to(device)
            self.labels = (aug_matrix * labels).t().flatten()

            features = handle_trainDataset(text_a, text_b, self.pipelines)
        else:
            self.labels = data[4].to(device)
            features = handle_evalDataset(text_a, text_b, self.pipelines)
        input_ids, segment_ids, soft_position, token_count, input_mask = features[0], features[1], features[2], features[3], features[4]
        input_ids = input_ids.to(device)
        segment_ids = segment_ids.to(device)
        soft_position = soft_position.to(device)
        token_count = token_count.to(device)
        input_mask = input_mask.to(device)

        outputs = self.bert_model(state, input_ids, segment_ids, soft_position, token_count, input_mask)
        text_embedding = outputs[:, 0]

        # 单独评估 P_BERT_SAM
        logits = self.y_text(text_embedding)
        self.pred_y = self.softmax(logits)[:, 1].unsqueeze(-1)


    def lossFunc(self):
        loss_a = F.binary_cross_entropy(input=self.pred_y, target=self.labels.unsqueeze(-1).float(), reduction='none')
        return loss_a.mean()
