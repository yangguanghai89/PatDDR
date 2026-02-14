import torch
import transformers

import utils, os
import random
import P_BERT.model_t_balance_no_new_for_PBERT_sample as model_t_balance_no
#import bigbird.model_t_balance_no_new_for_bigbird as model_t_balance_no
#import bigbird.model_t_balance_no_new_for_bigbird as model_t_balance_no

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from torch.utils.tensorboard import SummaryWriter

transformers.logging.set_verbosity_error()

main_dir = '/home/wangfei/study/dataset/'
# 加载BERT模型和分词器
model_dir = main_dir + 'wangfei/model/'
#bert base
tokenizer = BertTokenizer.from_pretrained(model_dir + 'uncased_L-12_H-768_A-12')              #bert-large-uncased   bert-base-uncased
bert_model = BertModel.from_pretrained(model_dir + 'uncased_L-12_H-768_A-12')

#bigbird
#tokenizer = BigBirdTokenizer.from_pretrained(model_dir + 'bigbird')
#bert_model = BigBirdModel.from_pretrained(model_dir + 'bigbird')

#longformer
#tokenizer = AutoTokenizer.from_pretrained(model_dir + "longformer")
#bert_model = LongformerModel.from_pretrained(model_dir + "longformer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(train = True, vaild = True, path =None):
    data_dir = main_dir + 'wangfei/dataset/'
    Tfile = data_dir + 'V6_t_inv_app_4/train_index.tsv'
    Vfile = data_dir + 'V6_t_inv_app_4/dev_index.tsv'
    train_df = pd.read_csv(Tfile, sep='\t')
    vaild_df = pd.read_csv(Vfile, sep='\t')
    train_data = utils.load_train_data(train_df)
    vaild_data = utils.load_train_data(vaild_df)

    # 显示数据大小
    train_data_size = len(train_data)
    vaild_data_size = len(vaild_data)
    print('训练集的大小是{}'.format(train_data_size))
    print('验证集的大小是{}'.format(vaild_data_size))

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 训练集
    train_batch_size = 16
    valid_batch_size = 8
    train_dataloader = DataLoader(train_data, batch_size = train_batch_size, shuffle = True)
    valid_dataloader = DataLoader(vaild_data, batch_size = valid_batch_size, shuffle = False)

    #初始化模型
    p_ipm = 0.5
    net = model_t_balance_no.net(bert_model, tokenizer, p_ipm).to(device)

    #优化器
    learning_rate = 0.00002
    optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate, weight_decay = 0.0005)

    # 训练的轮数
    epoch = 10
    max_training_step   = epoch * train_data_size / train_batch_size
    max_evaluation_step = epoch * vaild_data_size / valid_batch_size

    LS2TS = {}   # Learning step to Training loss
    LS2ES = {}   # Evaling step to Evaling loss
    LS2TM = {}   # Learning step to Training Model
    LS2EM = {}   # Evaling step to Evaling Model

    MLS_training_loss = 0 # MinLearningStep
    MLS_evaling_loss = 0

    # 加入可视化
    writer = SummaryWriter("logs_train")

    # 训练步骤开始
    net.train()
    total_train_step = 0
    train_loss = 0

    for i in range(epoch):
        print("----------第{}轮训练开始----------".format(i + 1))
        if train:
            # text_a, text_b, patentA, patentB, label, t, index, bu, dalei, inv_flag, app_flag
            state = 1
            for data in train_dataloader:
                text_a = list(data[0])
                text_b = list(data[1])
                labels = data[4].to(device)
                sClass = data[5].to(device)
                index = data[6].to(device)
                section = data[7].to(device)
                bClass = data[8].to(device)
                inv_flag = data[9].to(device)
                app_flag = data[10].to(device)

                net(state, labels, text_a, text_b, sClass, bClass, section, index, inv_flag, app_flag, True)  # [wenben, IPC]

                loss = net.lossFunc()
                train_loss = train_loss + loss.item()

                # 优化器优化模型
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step = total_train_step + 1
                if total_train_step % 100 == 0:
                    # 验证
                    valid_loss, total_valid_step = eval(net, valid_dataloader)

                    print("......训练次数是:{} / {}".format(total_train_step, max_training_step))
                    LS2TS[total_train_step] = train_loss / total_train_step
                    LS2ES[total_train_step] = valid_loss / total_valid_step

                    print("   训练集平均损失值:{}".format(LS2TS[total_train_step]))
                    print("   验证集平均损失值:{}".format(LS2ES[total_train_step]))
                    writer.add_scalar("train_loss", LS2TS[total_train_step], total_train_step)
                    writer.add_scalar("valid_loss", LS2ES[total_train_step],  total_train_step)

                    if (MLS_training_loss == 0) or (LS2TS[MLS_training_loss] > LS2TS[total_train_step]):
                        if MLS_training_loss != 0:
                            os.remove(LS2TM[MLS_training_loss])   # 移除旧模型

                        MLS_training_loss = total_train_step
                        LS2TM[MLS_training_loss] = 'saveModels/{}_{}_{}_{}.pth'.format("train", total_train_step,
                                                                                           LS2TS[total_train_step],
                                                                                           LS2ES[total_train_step])
                        torch.save(net, LS2TM[MLS_training_loss])

                    if (MLS_evaling_loss == 0) or (LS2ES[MLS_evaling_loss] > LS2ES[total_train_step]):
                        if MLS_evaling_loss != 0:
                            os.remove(LS2EM[MLS_evaling_loss]) # 移除旧模型

                        MLS_evaling_loss = total_train_step
                        LS2EM[MLS_evaling_loss] = 'saveModels/{}_{}_{}_{}.pth'.format("eval", total_train_step,
                                                                            LS2TS[total_train_step],
                                                                            LS2ES[total_train_step])
                        torch.save(net, LS2EM[MLS_evaling_loss])


def eval(net, valid_dataloader):
    # 验证步骤开始
    net.eval()
    valid_loss = 0
    total_valid_step = 0
    state = 2
    with torch.no_grad():
        for data in valid_dataloader:
            text_a = list(data[0])
            text_b = list(data[1])
            labels = data[4].to(device)
            sClass = data[5].to(device)
            index = data[6].to(device)
            section = data[7].to(device)
            bClass = data[8].to(device)
            inv_flag = data[9].to(device)
            app_flag = data[10].to(device)

            net(state, labels, text_a, text_b, sClass, bClass, section, index, inv_flag, app_flag, True)
            loss = net.lossFunc()
            valid_loss = valid_loss + loss.item()

            total_valid_step = total_valid_step + 1

        return valid_loss, total_valid_step

if __name__ == '__main__':
    train()
