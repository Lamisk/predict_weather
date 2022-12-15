#%%
from typing import List
import torch
from Data import Data
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from model import LSTM,GRU
import torch.optim as optim
import torch.nn as nn
from metrics import MAPE
import matplotlib.pyplot as plt
from ultis import plot_performance
#%%
df = pd.DataFrame(pd.read_excel("./test_sample.xlsx"))
df.drop(['precip','date','weather'],1,inplace=True)

df.head()
#%%
model_name = "GRU"
state = torch.load(f"./save/{model_name}")
model = state['state_dict']
scaler = state['scaler']
in_features  = state['in_features']
pred_features = state['pred_features']
input_width = state['input_width']
target_width = state['target_width']
shift = state['shift']
df_norm = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
# %%
input = df_norm.loc[:,df_norm.columns.isin(in_features)].values[:input_width]
target = df_norm.loc[:,df_norm.columns.isin(pred_features)].values[input_width+shift-target_width:input_width+shift]
input = torch.as_tensor(input).view(1,input.shape[0],input.shape[1]).float()
target = torch.as_tensor(target).float()
print(input.shape)
print(target.shape)
# print(input)
# print(target)

pred = model(input)
print(pred)
print(target)
index_target = {}
for col_name in pred_features:
    index_target[col_name] = df_norm.columns.get_loc(col_name)

print(index_target)
index = index_target['temp-Avg']
scaler.min_, scaler.scale_ = scaler.min_[index],scaler.scale_[index]

rs = scaler.inverse_transform(pred.detach().reshape(-1,1))
print(rs)
print(scaler.inverse_transform(target.detach().reshape(-1,1)))
# %%
df.head(14)
# %%
