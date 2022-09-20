from turtle import pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y

    def __len__(self):
        return len(self.Y)


def collate_fn(batch):
    x, y = zip(*batch)
    x = np.array(x)
    y = np.array(y)

    x = torch.tensor(x).to(dtype=torch.float32)
    y = torch.tensor(y).to(dtype=torch.float32)
    return x, y


def get_set_and_loader(X, Y, batch_size=64, shuffle=True):
    dataset = StockDataset(X=X, Y=Y)

    if batch_size == 0:
        batch_size = len(dataset)

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        collate_fn=collate_fn)

    return dataset, loader


def get_data(self, file_path, time, x_scaler = None, y_scaler = None):
        df = pd.read_csv(file_path)
        df['time'] = pd.to_datetime(df['time'], dayfirst=True)
        df = df.sort_values(by=df.columns[0])
        df = df[(df['time'] > time['start']) & (df['time'] < time['finish'])]
        array = df[df.columns[1]].to_numpy()
        is_array_nan = pd.isnull(df[df.columns[1]].to_frame()).to_numpy().squeeze()

        flag = False
        start = 0
        x = np.empty(shape=(0, self.seq_len))
        y = np.empty(shape=(0, self.future))

        for i in range(len(array)):
            if flag == False:
                if is_array_nan[i] == False:
                    start = i
                    flag = True
            else:
                if is_array_nan[i] == True or i == array.shape[0] - 1:
                    flag = False
                    if i - start <= self.seq_len + self.future:
                        continue
                    temp_x, temp_y = self.get_sequence_data(array[start:i])
                    if temp_x.shape[0] != 0:
                        x = np.concatenate((x, temp_x), axis=0)
                        y = np.concatenate((y, temp_y), axis=0)
        if x_scaler is None :  
            x_scaler = MinMaxScaler()
        
        x_scaler.fit(x)
        x = x_scaler.transform(x)
        if y_scaler is None : 
            y_scaler = MinMaxScaler()
        y_scaler.fit(y)
        y = y_scaler.transform(y)
        return {
            'X': x,
            'Y': y,
            'x-scaler': x_scaler,
            'y-scaler': y_scaler
        }


