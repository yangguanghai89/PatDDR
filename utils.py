import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def split_dataset(path):

    df = pd.read_csv(path)

    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df,valid_df

class load_train_data (Dataset):

    def __init__(self,dataset):
        super(load_train_data ,self).__init__()
        self.dataset = dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, item):

        row = self.dataset.iloc[item]

        text_a = row['#1 String']
        text_b = row['#2 String']
        patentA = row['#1 ID']
        patentB = row['#2 ID']
        t = torch.tensor(row['SClass_Sim'], dtype = torch.float32)
        bu = torch.tensor(row['Section_Sim'], dtype = torch.float32)
        dalei = torch.tensor(row['BClass_Sim'], dtype=torch.float32)
        inv_flag = torch.tensor(row['Inv_Sim'], dtype=torch.float32)
        app_flag = torch.tensor(row['App_Sim'], dtype=torch.float32)
        label = torch.tensor(row['Quality'], dtype = torch.float32)
        index = torch.tensor(row['Index'], dtype = torch.long)

        # return (title_a, title_b, text_a, text_b, patentA, patentB, t, label, item)

        return (text_a, text_b, patentA, patentB, label, t, index, bu, dalei, inv_flag, app_flag)


class load_test_data (Dataset):

    def __init__(self,dataset):
        super(load_test_data ,self).__init__()
        self.dataset = dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, item):

        row = self.dataset.iloc[item]

        text_a = row['#1 String']
        text_b = row['#2 String']
        patentA = row['#1 ID']
        patentB = row['#2 ID']
        t = torch.tensor(int(row['SClass_Sim']), dtype=torch.float32)
        bu = torch.tensor(int(row['Section_Sim']), dtype=torch.float32)
        dalei = torch.tensor(int(row['BClass_Sim']), dtype=torch.float32)
        inv_flag = torch.tensor(int(row['Inv_Sim']), dtype=torch.float32)
        app_flag = torch.tensor(int(row['App_Sim']), dtype=torch.float32)
        label = torch.tensor(int(row['Quality']), dtype=torch.float32)
        index = torch.tensor(int(row['Index']), dtype=torch.long)

        # return (title_a, title_b, text_a, text_b, patentA, patentB, t, label, item)

        return (text_a, text_b, patentA, patentB, label, t, index, bu, dalei, inv_flag, app_flag)



# def sorting (combined): #排序
#     df = pd.DataFrame(combined, columns=['patentA', 'patentB', 'label', 'pred_y'])
#     df_sorted = df.sort_values(by='pred_y', ascending=False)
#     return df_sorted

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)