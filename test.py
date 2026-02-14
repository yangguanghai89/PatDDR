import torch, codecs
import utils
from evaluation import my_evalution
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import transformers

transformers.logging.set_verbosity_error()

main_dir = '/home/wangfei/study/dataset/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_csv_TOP():
    data_dir = main_dir + 'wangfei/dataset/'
    tfile = data_dir + 'V6_t_inv_app_4/test_index.tsv'
    reader = codecs.open(filename=tfile, mode='r', encoding='utf-8')
    header = reader.readline().strip().split('\t')

    data = []
    while True:
        line = reader.readline()
        if not line:
            break

        ss = line.strip().split('\t')
        data.append(ss)
        #if len(data) == 1000:
        #    yield (header, data)

        #    data.clear()
    return header, data

def train(path):
    header, data = read_csv_TOP()
    test_df = pd.DataFrame(data, columns=header)

    test_data = utils.load_test_data(test_df)

    test_data_size = len(test_data)
    print('测试集的大小是{}'.format(test_data_size))

    test_batch_size = 100
    test_dataloader = DataLoader(test_data, batch_size = test_batch_size, shuffle = False)

    #初始化模型
    net = torch.load(path, weights_only=False).to(device)

    net.eval()
    print("----------测试开始----------")

    lst = []
    my_result = {}
    state = 3
    with torch.no_grad():
        for data in test_dataloader:

            patentA = data[2]
            patentB = data[3]
            labels = data[4].to(device)

            net(state, data)

            pred_y = net.pred_y

            combined = np.concatenate((np.expand_dims(np.array(patentA), axis=1),
                                                   np.expand_dims(np.array(patentB), axis=1),
                                                   labels.unsqueeze(1).to('cpu').numpy(),
                                                   pred_y.to('cpu').numpy()), axis=1)
            lst.extend(combined)
            if len(lst) == 1000:
                result_list = []
                result_list.extend(my_evalution.mergeResult(tid=lst[0][0],
                                                            sids=[row[1] for row in lst],
                                                            labels=[row[2] for row in lst],
                                                            weights=[row[3] for row in lst]))
                my_result[lst[0][0]] = result_list
                lst.clear()

        actual = my_evalution.readQRELS(main_dir + 'wangfei/dataset/test_qrels.txt')
        my_evalution.evalute(my_result, actual)

def evaluate_full():
    import os
    eval_result_dir = '/home/wangfei/study/wangfei/python/pyproject08/new_net/saveTemp/'
    fnames = os.listdir(eval_result_dir)

    result = {}
    num = 0
    for fname in fnames:
        if not fname.startswith('EP'):
            continue

        reader = codecs.open(filename=eval_result_dir+fname, mode='r', encoding='utf-8')
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
    QRELS = my_evalution.readQRELS(main_dir + 'wangfei/dataset/test_qrels.txt')

    # 进行数据评估
    my_evalution.evalute(result, QRELS)

if __name__ == '__main__':
    #train('saveModels/eval_1700_0.07427471539293251_0.062193602340563925.pth') eval_500_0.12565589665621518_0.06283277854730654
    train('saveModels/eval_1100_0.08945568282432347_0.06250355212943227.pth')
    #evaluate_full()






