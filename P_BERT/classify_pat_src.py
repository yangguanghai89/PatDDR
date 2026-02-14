# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Fine-tuning on A Classification Task with pretrained Transformer """

import itertools
import csv
import random

import fire, codecs

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import tokenization
import models
import optim
import train, os, pickle

from P_BERT.utils import set_seeds, get_device, truncate_tokens_pair
from evaluation.my_evalution import mergeResult, readQRELS, evalute

def readTestPair(file):
    reader = codecs.open(filename=file, mode='r', encoding='utf-8')
    reader.readline() # 跳过首句
    tids = list()
    sids = list()
    labels = list()
    textpairs = list()
    while True:
        line = reader.readline()
        if not line:
            break
        ss = line.strip().split('\t')
        tids.append(ss[1])
        sids.append(ss[2])
        labels.append(ss[5])
        textpairs.append((ss[5], ss[3], ss[4]))

        if len(tids) == 1000:
            yield tids, sids, labels, textpairs

            tids = list()
            sids = list()
            labels = list()
            textpairs = list()
    reader.close()
    return None, None, None, None

def readTestPair_SRC(file):
    reader = codecs.open(filename=file, mode='r', encoding='utf-8')
    reader.readline() # 跳过首句
    tids = list()
    sids = list()
    labels = list()
    textpairs = list()
    while True:
        line = reader.readline()
        if not line:
            break
        ss = line.strip().split('\t')
        tids.append(ss[1])
        sids.append(ss[2])
        labels.append(ss[9])
        textpairs.append((ss[9], ss[3]+' '+ss[5], ss[4]+' '+ss[6]))

        if len(tids) == 1000:
            yield tids, sids, labels, textpairs

            tids = list()
            sids = list()
            labels = list()
            textpairs = list()
    reader.close()
    return None, None, None, None

class CsvDataset(Dataset):
    """ Dataset Class for CSV file """
    labels = None
    def __init__(self, state, file, data_slice, pipeline=[]): # cvs file and pipeline object
        Dataset.__init__(self)
        if state==1:
            self.handle_trainDataset(file, pipeline)
        elif state==2:
            self.handle_evalDataset(file, pipeline)
        elif state==3:
            self.handle_testDataset(data_slice, pipeline)
        else:
            pass

    def handle_trainDataset(self, file, pipeline):
        data = []
        with open(file, "r", encoding='utf-8') as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance, ratio in self.get_instances(lines):  # instance : tuple of fields
                if ratio == 1:
                    for proc in pipeline:  # a bunch of pre-processing
                        instance = proc(instance)
                    data.append(instance)
                else:
                    # 数据增强
                    instances = self.dataAugmentation2(instance, ratio=ratio)
                    for instance in instances:
                        for proc in pipeline:  # a bunch of pre-processing
                            instance = proc(instance)
                        data.append(instance)

        print('Train example:' + str(len(data)))

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def handle_evalDataset(self, file, pipeline):
        data = []
        with open(file, "r", encoding='utf-8') as f:
            # list of splitted lines : line is also list
            lines = csv.reader(f, delimiter='\t', quotechar=None)
            for instance, ratio in self.get_instances(lines):  # instance : tuple of fields
                for proc in pipeline:  # a bunch of pre-processing
                    instance = proc(instance)
                data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def handle_testDataset(self, data_slice, pipeline):
        Dataset.__init__(self)
        data = []
        for instance in data_slice:
            for proc in pipeline:
                instance = proc(instance)
            data.append(instance)

        # To Tensors
        self.tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*data)]

    def dataAugmentation(self, instance, ratio=2):
        instances = list()
        label, text_a, text_b = instance
        text_a = text_a.split(' ')
        text_b = text_b.split(' ')
        val = min(len(text_a), len(text_b))
        step = val // ratio
        for i in range(ratio):
            if i == 0:
                instances.append(instance)
            else:
                text_a = text_a[i*step:len(text_a)] +text_a[0:i*step]
                text_b = text_b[i * step:len(text_b)] + text_b[0:i * step]
                instances.append((label, ' '.join(text_a), ' '.join(text_b)))
        return instances
    
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

    def __len__(self):
        return self.tensors[0].size(0)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def get_serialize_data(self):
        return self.tensors

    def reverse_serialize(self, tensors):
        self.tensors = tensors

    def get_instances(self, lines):
        """ get instance array from (csv-separated) line list """
        raise NotImplementedError



class MRPC(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, bTest, file, data_slice, pipeline=[]):
        super().__init__(bTest, file, data_slice, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b


class MNLI(CsvDataset):
    """ Dataset class for MNLI """
    labels = ("contradiction", "entailment", "neutral") # label names
    def __init__(self, bTest, file, data_slice, pipeline=[]):
        super().__init__(bTest, file, data_slice, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[-1], line[8], line[9] # label, text_a, text_b

class PAT(CsvDataset):
    """ Dataset class for MRPC """
    labels = ("0", "1") # label names
    def __init__(self, state, file, data_slice, pipeline=[]):
        super().__init__(state, file, data_slice, pipeline)

    def get_instances(self, lines):
        for line in itertools.islice(lines, 1, None): # skip header
            yield line[0], line[3], line[4] # label, text_a, text_b

class PAT_SRC(CsvDataset):
    """ Dataset class for PAT """
    labels = ("0", "1") # label names
    def __init__(self, bTest, file, data_slice, pipeline=[]):
        super().__init__(bTest, file, data_slice, pipeline)
        ratio = {'A':2, 'B':1, 'C':1, 'D':16, 'E':32, 'F':2, 'G':1, 'H':1}  # 第一轮增强


    def get_instances(self, lines):
        #ratios = {'A': 2, 'B': 1, 'C': 1, 'D': 2, 'E': 2, 'F': 2, 'G': 1, 'H': 1}  # 第一轮增强
        #ratios = {'A': 2, 'B': 1, 'C': 1, 'D': 25, 'E': 45, 'F': 4, 'G': 1, 'H': 1}  # 第一轮增强
        ratios = {'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 'H': 1}  # 第一轮增强
        for line in itertools.islice(lines, 1, None): # skip header
            #flag = line[7][0]
            #ratio = ratios[flag]
            yield (line[9], line[3] + ' ' + line[5], line[4] + ' ' + line[6]), 1       # label, text_a, text_b, ratio

def dataset_class(task):
    """ Mapping from task string to Dataset Class """
    table = {'mrpc': MRPC, 'mnli': MNLI, 'pat':PAT, 'pat_src':PAT_SRC}
    return table[task]

class Pipeline():
    """ Preprocess Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Tokenizing(Pipeline):
    """ Tokenizing sentence pair """
    def __init__(self, preprocessor, tokenize):
        super().__init__()
        self.preprocessor = preprocessor # e.g. text normalization
        self.tokenize = tokenize # tokenize function

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        tokens_a = self.tokenize(self.preprocessor(text_a))
        tokens_b = self.tokenize(self.preprocessor(text_b)) \
                   if text_b else []

        return (label, tokens_a, tokens_b)

class Tokenizing_with_position_count_for_prefixSPE(Pipeline):
    ''' Tokening sentence pair '''
    def __init__(self, tokenizer):
        super().__init__()
        self.preprocessor = tokenizer.convert_to_unicode  # e.g. text normalization
        self.tokenize = tokenizer.tokenize  # tokenize function
        self.basic_tokenize = tokenizer.basic_tokenizer.tokenize

    def __call__(self, instance):
        label, text_a, text_b = instance

        label = self.preprocessor(label)
        token_a, soft_position_a, token_count_a = self.tokenize_with_position_count(text_a)
        token_b, soft_position_b, token_count_b = self.tokenize_with_position_count(text_b)

        return (label, token_a, soft_position_a, token_count_a, token_b, soft_position_b, token_count_b)


    def tokenize_with_position_count(self, full_sequnece):
        term2count = {}
        for term in self.basic_tokenize(full_sequnece):
            if term not in term2count.keys():  #
                term2count[term] = 0
            term2count[term] += 1

        split_tokens = []
        soft_position = []
        token_count = []

        term2einx = {}
        preterm = None

        for term in self.basic_tokenize(full_sequnece):

            if term in term2einx.keys():  # 不记录重复项
                preterm = term
                continue

            if preterm == None:
                preterm_inx = 0
            else:
                preterm_inx = term2einx[preterm]
            cur_inx = preterm_inx
            for token in self.tokenize(term):
                split_tokens.append(token)
                token_count.append(term2count[term])
                cur_inx += 1
                soft_position.append(cur_inx)

            term2einx[term] = cur_inx  # 信息归档
            preterm = term

        return split_tokens, soft_position, token_count


class AddSpecialTokensWithTruncation(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2
        truncate_tokens_pair(tokens_a, tokens_b, _max_len)

        # Add Special Tokens
        tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        tokens_b = tokens_b + ['[SEP]'] if tokens_b else []

        return (label, tokens_a, tokens_b)

class AddSpecialTokensWithTruncation2(Pipeline):
    """ Add special tokens [CLS], [SEP] with truncation """
    def __init__(self, max_len=512):
        super().__init__()
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, soft_position_a, token_count_a, tokens_b, soft_position_b, token_count_b = instance

        # -3 special tokens for [CLS] text_a [SEP] text_b [SEP]
        # -2 special tokens for [CLS] text_a [SEP]
        _max_len = self.max_len - 3 if tokens_b else self.max_len - 2

        truncate_tokens_pair(tokens_a, tokens_b, _max_len)
        truncate_tokens_pair(soft_position_a, soft_position_b, _max_len)
        truncate_tokens_pair(token_count_a, token_count_b, _max_len)

        tokens = []
        segment_ids = []
        soft_position = []
        token_count = []

        max_position_a = 0

        tokens.append("[CLS]")
        segment_ids.append(0)
        soft_position.append(0)
        token_count.append(1)
        for i, token in enumerate(tokens_a):
            tokens.append(token)
            segment_ids.append(0)
            soft_position.append(soft_position_a[i])
            max_position_a = soft_position_a[i] if soft_position_a[i] > max_position_a else max_position_a
            token_count.append(token_count_a[i])
        tokens.append("[SEP]")
        segment_ids.append(0)
        max_position_a += 1
        soft_position.append(max_position_a)
        token_count.append(2)

        max_position_b = max_position_a
        if tokens_b:
            for i, token in enumerate(tokens_b):
                tokens.append(token)
                segment_ids.append(1)
                soft_position.append(soft_position_b[i] + max_position_a)  # 确保soft_position_b的位置与 soft_position_a不重合
                max_position_b = (soft_position_b[i] + max_position_a) if (soft_position_b[i] + max_position_a) > max_position_b else max_position_b
                token_count.append(token_count_b[i])
            tokens.append("[SEP]")
            segment_ids.append(1)
            max_position_b += 1
            soft_position.append(max_position_b)
            token_count.append(2)

            # 如果某一单词的数目高于128,则直接设置为127
            for i, tc in enumerate(token_count):
                if tc >= self.max_len:
                    # print(tokens_a)
                    # print(tokens_b)
                    # print("Error: 字符{}数目为{}！".format(tokens[i], tc))
                    token_count[i] = self.max_len - 1
                    # exit(0)

        return (label, tokens, segment_ids, soft_position, token_count)



class TokenIndexing(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens_a, tokens_b = instance

        input_ids = self.indexer(tokens_a + tokens_b)
        segment_ids = [0]*len(tokens_a) + [1]*len(tokens_b) # token type ids
        input_mask = [1]*(len(tokens_a) + len(tokens_b))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, label_id)


class TokenIndexing2(Pipeline):
    """ Convert tokens into token indexes and do zero-padding """
    def __init__(self, indexer, labels, max_len=512):
        super().__init__()
        self.indexer = indexer # function : tokens to indexes
        # map from a label name to a label index
        self.label_map = {name: i for i, name in enumerate(labels)}
        self.max_len = max_len

    def __call__(self, instance):
        label, tokens, segment_ids, soft_position, token_count = instance

        input_ids = self.indexer(tokens)
        input_mask = [1]*(len(tokens))

        label_id = self.label_map[label]

        # zero padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        soft_position.extend([0] * n_pad)
        token_count.extend([0] * n_pad)
        input_mask.extend([0]*n_pad)

        return (input_ids, segment_ids, soft_position, token_count, input_mask, label_id)


class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, state, input_ids, segment_ids, soft_position, token_count, input_mask):
        h = self.transformer(state, input_ids, segment_ids, soft_position, token_count, input_mask)
        # only use the first h in the sequence
        '''
        len = h.size()[1]
        h = torch.sum(h, dim=1)
        h = h / len
        pooled_h = self.activ(self.fc(h))
        '''

        pooled_h = self.activ(self.fc(h[:, 0]))

        if state == 1:   #训练
            logits = self.classifier(self.drop(pooled_h))
        else:
            logits = self.classifier(pooled_h)
        return logits


#pretrain_file='../uncased_L-12_H-768_A-12/bert_model.ckpt',
#pretrain_file='../exp/bert/pretrain_100k/model_epoch_3_steps_9732.pt',

def saveEvalResultBasedAcc(save_file, model_acc):
    writer = codecs.open(filename=save_file, mode='w', encoding='utf-8')
    max_name = ''
    max_acc = 0.0
    ostr = ''
    model_acc = sorted(model_acc.items(), key=lambda x:x[0])
    for id,(name, acc) in model_acc:
        ostr += 'all_model_checkpoint_path:' + name + ', ' + str(acc) + '\n'
        if acc > max_acc:
            max_name = name
            max_acc = acc
    writer.write('model_checkpoint_path:' + max_name + ',' + str(max_acc) + '\n')
    writer.write(ostr)
    writer.flush()
    writer.close()

def saveEvalResultBasedLoss(save_file, model_loss):
    writer = codecs.open(filename=save_file, mode='w', encoding='utf-8')
    min_name = ''
    min_loss = 1000.0
    ostr = ''
    model_loss = sorted(model_loss.items(), key=lambda x:x[0], reverse=False)
    for id,(name, loss) in model_loss:
        ostr += 'all_model_checkpoint_path:' + name + ', ' + str(loss) + '\n'
        if loss < min_loss:
            min_name = name
            min_loss = loss
    writer.write('model_checkpoint_path:' + min_name + ',' + str(min_loss) + '\n')
    writer.write(ostr)
    writer.flush()
    writer.close()

def readBestModelName(save_file):
    reader = codecs.open(filename=save_file, mode='r', encoding='utf-8')
    line = reader.readline()
    ss = line.split(':')
    ss = ss[1].split(',')
    bestModelName = ss[0]
    reader.close()
    return bestModelName

def data_serialization(data_dir, TaskDataset, state, pipeline, maxlen):

    serial_dir = data_dir + 'test_serial_aug_exa_' + str(maxlen) + '/'
    if os.path.exists(serial_dir):
        return serial_dir
    else:
        os.mkdir(serial_dir)

    for tids, sids, labels, dataset in readTestPair_SRC(data_dir + 'test.tsv'):
        tensors = TaskDataset(state, None, dataset, pipeline).get_serialize_data()
        with open(serial_dir+tids[0]+'.pkl', 'wb') as file:
            pickle.dump((tids, sids, labels, tensors), file)

    return serial_dir

def main(task='pat_src',
         btrain=True,
         beval=False,
         btest=False,
         train_cfg='config/train_pat.json',
         model_cfg='config/bert_base.json',
         #model_cfg='config/scibert_base.json',
         #data_dir='/home/wangfei/study/source/dataset/V6/',
         data_dir='/home/wangfei/study/source/dataset/V6_SRC_IPC/',
         #model_file='tmp/pat_output/model_steps_933.pt',            # 制定模型
         model_file='tmp/pat_output_128/model_steps_1200.pt',                               # 从评估文件读取
         pretrain_file='/home/wangfei/study/source/model/BERT_BASED_DIR/tensorflow/bert_model.ckpt',
         #pretrain_file='/home/wangfei/study/source/model/scibert_scivocab_uncased/bert_model.ckpt',
         data_parallel=True,
         vocab='/home/wangfei/study/source/model/BERT_BASED_DIR/tensorflow/vocab.txt',
         #vocab='/home/wangfei/study/source/model/scibert_scivocab_uncased/vocab.txt',
         save_dir='tmp/pat_output',
         max_len=128):

    save_dir = save_dir + '_' + str(max_len) + '/'

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    TaskDataset = dataset_class(task) # task dataset class according to the task

    pipeline = [Tokenizing_with_position_count_for_prefixSPE(tokenizer),
                AddSpecialTokensWithTruncation2(max_len),
                TokenIndexing2(tokenizer.convert_tokens_to_ids,
                              TaskDataset.labels, max_len)]
    '''
    pipeline = [Tokenizing_with_position_count_for_prefixSPE(tokenizer)]
    '''

    model = Classifier(model_cfg, len(TaskDataset.labels))
    criterion = nn.CrossEntropyLoss()
    #adam = torch.optim.Adam(model.parameters(), lr=0.02, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0005)

    trainer = train.Trainer(cfg,
                            model,
                            None,
                            optim.optim4GPU(cfg, model),
                            #adam,
                            save_dir, get_device())

    def get_loss(state, model, batch, global_step=None):  # make sure loss is a scalar tensor
        input_ids, segment_ids, soft_position, token_count, input_mask, label_id = batch
        logits = model(state, input_ids, segment_ids, soft_position, token_count, input_mask)
        loss = criterion(logits, label_id)
        return loss

    if btrain == True:

        state = 1
        dataset = TaskDataset(state, data_dir+'train.tsv', None, pipeline)
        data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        trainer.train(state, get_loss, data_iter, None, pretrain_file, data_parallel)

    if beval == True:
        def evaluate(model, batch):
            input_ids, segment_ids, soft_position, token_count, input_mask, label_id = batch
            logits = model(input_ids, segment_ids, soft_position, token_count, input_mask)
            _, label_pred = logits.max(1)
            result = (label_pred == label_id).float() #.cpu().numpy()
            accuracy = result.mean()
            return accuracy, result

        if model_file != None:
            state=2
            dataset = TaskDataset(state, data_dir + 'dev.tsv', None, pipeline)
            data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

            results = trainer.eval_loss(state, get_loss, data_iter, model_file, data_parallel)
            #total_loss = torch.cat(results).mean().item()
            total_loss = 0.0
            for loss in results:
                total_loss += loss
            loss = (total_loss / len(results)).item()
            print('loss:', loss)

        else:
            state=2
            dataset = TaskDataset(state, data_dir + 'dev.tsv', None, pipeline)

            names = os.listdir(save_dir)
            #model_acc = {}
            model_loss = {}
            for name in names:
                if not name.startswith('model_steps'):
                    continue

                id = name[12:name.index('.')]
                if not name.startswith('model_steps'):
                    continue

                data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

                results = trainer.eval_loss(state, get_loss, data_iter, save_dir + '/' + name, data_parallel)
                #loss = torch.cat(results).mean().item()
                total_loss = 0.0
                for loss in results:
                    total_loss += loss
                loss = (total_loss / len(results)).item()
                model_loss[int(id)] = tuple([name, loss])
                print(name, '（loss）:', loss)

            #saveEvalResultBasedAcc(save_dir + '/eval_result', model_acc)
            saveEvalResultBasedLoss(save_dir + '/eval_result', model_loss)

    if btest == True:

        state = 3
        serial_file = data_serialization(data_dir, TaskDataset, state, pipeline, max_len)

        def predict(state, model, batch):
            input_ids, segment_ids, soft_position, token_count, input_mask, label_id = batch
            logits = model(state, input_ids, segment_ids, soft_position, token_count, input_mask)
            logits_softmax = F.softmax(logits, dim=1)
            label_pred = logits_softmax[:, 1]
            return label_pred

        '''
        if model_file == None:   # 未制定 则读取最优评估结果
            model_file = save_dir + '/' + readBestModelName(save_dir + '/eval_result')
        '''
        model_dir = 'tmp/pat_output_' + str(max_len) + '/'
        result_dir = 'saveTemp/saveTemp_' + str(max_len) + '/'
        model_files = []
        save_dirs = []
        #for i in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
        for i in [311, 622, 933, 1244, 1555, 1866]:
            model_files.append(model_dir + 'model_steps_{}.pt'.format(i))
            save_dirs.append(result_dir + 'saveTemp_pat_{}/'.format(i))
        '''
        model_dir = 'tmp/pat_output_' + str(max_len) + '/'
        model_files = [model_dir + 'model_steps_622.pt', model_dir+'model_steps_1244.pt', model_dir+'model_steps_1866.pt', model_dir+'model_steps_2488.pt', model_dir+'model_steps_3110.pt', model_dir+'model_steps_3732.pt']
        #model_files = [model_dir + 'model_steps_1555.pt', model_dir + 'model_steps_1866.pt']
        result_dir = 'saveTemp/saveTemp_' + str(max_len) + '/'
        save_dirs   = [result_dir+'saveTemp_pat_622/', result_dir+'saveTemp_pat_1244/', result_dir+'saveTemp_pat_1866/', result_dir+'saveTemp_pat_2488/', result_dir+'saveTemp_pat_3110/', result_dir+'saveTemp_pat_3732/']
        #save_dirs = [result_dir + 'saveTemp_pat_1555/', result_dir + 'saveTemp_pat_1866/']
        '''
        #model_files = []
        #model_files.append(model_file)
        #save_dirs = ['saveTemp/saveTemp/']
        save_dir = 'saveTemp/saveTemp/'

        serial_fnames = os.listdir(serial_file)  # 获取pkl文件
        serial_fnames.sort()
        for i in range(1):
            trainer.load(model_file=model_file, pretrain_file=None)
            results = {}
            for fname in serial_fnames:
                with open(file=serial_file+fname, mode='rb') as file:
                    (tids, sids, labels, tensors) = pickle.load(file)
                    dataset = TaskDataset(4, None, None, None)        #state=4,表示直接加载pickle数据
                    dataset.reverse_serialize(tensors)
                    data_iter = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
                    weights = trainer.predict_without_load_model(state, predict, data_iter, data_parallel)
                    results[tids[0]] = mergeResult(tids[0], sids, labels, weights, topN=1000, bSave=True,
                                                   save_dir=save_dir)

            # 读取评估数据
            test_dir = '/home/wangfei/study/source/dataset/'
            qrels = test_dir + 'test_qrels.txt'
            QRELS = readQRELS(qrels)

            # 进行数据评估
            evalute(results, QRELS, save_dir=save_dir)


def evaluate_full():
    dir = 'saveTemp/saveTemp_pat_311/'
    fnames = os.listdir(dir)

    result = {}
    num = 0
    for fname in fnames:
        if not fname.startswith('EP'):
            continue

        reader = codecs.open(filename=dir+fname, mode='r', encoding='utf-8')
        sids = []
        while True:
            line = reader.readline()
            if len(line) == 0:
                break

            ss = line.strip().split('\t')
            sids.append(ss[1])
        result[fname] = sids
        reader.close()
        num += 1

    print('共读取主题专利数目：' + str(num))

    # 读取评估数据
    data_dir = '/home/wangfei/study/source/dataset/'
    #qrels = FLAGS.data_dir + '/test_qrels.txt'
    qrels = data_dir + 'test_qrels.txt'
    QRELS = readQRELS(qrels)

    # 进行数据评估
    evalute(result, QRELS)

if __name__ == '__main__':
    fire.Fire(main)

    #evaluate_full()
