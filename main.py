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
#%%
df = pd.DataFrame(pd.read_excel("./data/cras_weather_final_2021.xlsx"))
df.drop('precip',1,inplace=True)
df.info()

# %%
df.describe().T
#%%
input_width = 12
target_width = 1
shift = 1
data = Data(df=df,input_width=input_width,target_width=target_width,shift=shift)
data.split_df()
print(data.df_train.shape)
print(data.df_valid.shape)
print(data.df_eval.shape)
data.df_train.head()
# %%

scaler = MinMaxScaler()
data.apply_scaler(scaler)
data.df_train_norm.head()
# %%
data.heat_map()
#%%
in_features = ['temp-Max','dew-point-Avg','wind-Max','temp-Avg']
pred_features = ['temp-Avg']
data.gen_dataloader(in_features,pred_features)
# %%
input_dim = len(in_features)
output_dim = len(pred_features)


lstm = LSTM(input_dim=input_dim,hidden_dim=50,output_dim=output_dim,num_layers=1)
gru  = GRU (input_dim=input_dim,hidden_dim=50,output_dim=output_dim,num_layers=1)

dict_models = {
    "LSTM": lstm,
    "GRU": gru
}

metric_valid = []
metric_eval  = []

for model_name in dict_models:
    lr = 1e-3
    weight_decay = 1e-6
    epochs = 300
    loss_fn = nn.MSELoss(reduction='mean')
    metric = MAPE
    model = dict_models[model_name]
    optimizer = optim.Adam(lr=lr,weight_decay=weight_decay,params=model.parameters()) 
    data.train(epochs = epochs, model=model,loss_fn=loss_fn,optim=optimizer,metric=metric)
    data.plot_eval_data(['temp-Avg','dew-point-Avg'],1)
    data.plot_all_eval_data(['temp-Avg'])
    metric_valid.append(torch.mean(torch.as_tensor(data.hist_metric_valid)))
    metric_eval.append(torch.mean(torch.as_tensor(data.hist_metric_eval)))
    state = {
        'state_dict': data.model,
        'scaler': data.scaler,
        'in_features':in_features,
        'pred_features': pred_features,
        'input_width': input_width,
        'target_width':target_width,
        'shift':shift

    }
    torch.save(state,f'save/{model_name}')

#%%

plot_performance(dict_models,['temp-Avg'],metric_valid,metric_eval,metric)
# %%

