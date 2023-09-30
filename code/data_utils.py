import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


train = pd.read_csv('../dataset/train.csv')
test = pd.read_csv('../dataset/test.csv')
submission = pd.read_csv('../dataset/sample_submission.csv')
submission['SalePrice'] = 0

y_train = np.log1p(train['SalePrice'])

train = train.drop(columns = ['Id', 'SalePrice'])
test = test.drop(columns = ['Id'])

numeric = train.dtypes[train.dtypes != object].index
train[numeric] = train[numeric].fillna(0)
test[numeric] = test[numeric].fillna(0)

scaler = StandardScaler()
train[numeric] = scaler.fit_transform(train[numeric])
test[numeric] = scaler.transform(test[numeric])

data = pd.concat([train, test], axis = 0)
data = pd.get_dummies(data, dummy_na = True, drop_first = True)

x_train = data.iloc[:1460]
x_test = data.iloc[1460:]

#%%
class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x_data = x.values
        self.y_data = y.values.reshape(len(y), -1)
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y