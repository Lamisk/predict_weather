from typing import Dict, List
import pandas as pd
import torch
from torch.functional import Tensor
from torch.utils.data import TensorDataset, DataLoader, dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
from tqdm import tqdm

class Data():
    def __init__(self,df:pd.DataFrame=None,input_width=12,target_width=1,shift=1) -> None:
        assert shift>=target_width
        self.df = df

        self.hist_loss_train = []
        self.hist_loss_valid = []
        self.hist_metric_valid = []
        self.hist_metric_eval = []

        self.scaler = None
        self.model = None
        self.loss_fn = None
        self.epochs:int = 10
        self.optim = None
        self.batch_size:int = 4
        self.metric = None

        self.dataloader_train:DataLoader = None
        self.dataloader_valid:DataLoader = None
        self.dataloader_eval:DataLoader  = None

        self.df_train:pd.DataFrame = None
        self.df_valid:pd.DataFrame = None
        self.df_eval:pd.DataFrame  = None

        self.df_train_norm:pd.DataFrame = None
        self.df_valid_norm:pd.DataFrame = None
        self.df_eval_norm:pd.DataFrame  = None

        self.date_time = pd.to_datetime(self.df.pop('date'),format='%d/%m/%Y')
        self.weather_status = self.df.pop('weather')

        self.input_width:int  = input_width
        self.target_width:int = target_width
        self.shift:int        = shift
        self.window_width:int = self.input_width + self.shift

        self.inputs_train:Tensor  = None
        self.targets_train:Tensor = None
        self.inputs_valid:Tensor  = None
        self.targets_valid:Tensor = None

        self.index_of_eval_input_column  = None
        self.index_of_eval_target_column = None

        self.input_slice = slice(0, input_width)
        self.input_indices = torch.arange(self.window_width)[self.input_slice]

        self.target_start = self.window_width - self.target_width
        self.targets_slice = slice(self.target_start, None)
        self.target_indices = torch.arange(self.window_width)[self.targets_slice]

        self.inputs_columns_names = None
        self.targets_columns_names = None

        

    def gen_dataloader(self,inputs_columns_names:List[str],targets_columns_names:List[str]):
        assert [tc in inputs_columns_names for tc in targets_columns_names].count(False)==0, "targets_columns_names must be in inputs_columns_names" 
        self.inputs_train,self.targets_train = self.gen_sequence(self.df_train_norm,inputs_columns_names,targets_columns_names)
        self.inputs_valid,self.targets_valid = self.gen_sequence(self.df_valid_norm,inputs_columns_names,targets_columns_names)
        self.inputs_eval,self.targets_eval   = self.gen_sequence(self.df_eval_norm,inputs_columns_names,targets_columns_names)

        dataset_train = TensorDataset(self.inputs_train,self.targets_train)
        dataset_valid = TensorDataset(self.inputs_valid,self.targets_valid)
        dataset_eval  = TensorDataset(self. inputs_eval,self.targets_eval)
        

        self.dataloader_train = DataLoader(dataset = dataset_train,batch_size=self.batch_size,shuffle=False)
        self.dataloader_valid = DataLoader(dataset = dataset_valid,batch_size=self.batch_size,shuffle=False)
        self.dataloader_eval  = DataLoader(dataset = dataset_eval, batch_size=1,shuffle=False)
        
        indices_input_of = {col_name:self.df_train.columns.get_loc(col_name) for col_name in inputs_columns_names }
        indices_input_of = sorted(indices_input_of.items(), key=lambda item: item[1])
        self.index_of_eval_input_column = {k: idx for idx,(k, v) in enumerate(indices_input_of)}

        indices_target_of = {col_name:self.df_train.columns.get_loc(col_name) for col_name in targets_columns_names }
        indices_target_of = sorted(indices_target_of.items(), key=lambda item: item[1])
        self.index_of_eval_target_column = {k: idx for idx,(k, v) in enumerate(indices_target_of)}

       

    def gen_sequence(self,df:pd.DataFrame,inputs_columns_names:List[str],targets_columns_names:List[str]):
        inputs = []
        targets = []
        self.inputs_columns_names = inputs_columns_names
        self.targets_columns_names = targets_columns_names
        for index in range(df.shape[0]):
            if index + self.input_width -1 >= df.shape[0]:
                break
            if index+self.input_width -1+self.shift >= df.shape[0]:
                break
            if index + self.input_width +self.shift -self.target_width >=df.shape[0]:
                break

            input = df.loc[index:index + self.input_width -1 ,df.columns.isin(inputs_columns_names)].values
            target = df.loc[index + self.input_width +self.shift -self.target_width :index+self.input_width -1+self.shift   ,df.columns.isin(targets_columns_names)].values
            inputs.append(input)
            targets.append(target)
           
        return  Variable(torch.as_tensor(inputs).float(),requires_grad=True),Variable(torch.as_tensor(targets).float(),requires_grad=True)

        
    def split_df(self,prob_train:float=0.7,prob_valid:float=0.2,prob_eval:float=0.1):
        num_rows = self.df.shape[0]
        self.df_train = self.df[0:int(num_rows*prob_train)]
        self.df_valid = self.df[int(num_rows*prob_train):int(num_rows*(prob_train+prob_valid))]
        self.df_eval = self.df[int(num_rows*(prob_train+prob_valid)):int(num_rows*(prob_train+prob_valid+prob_eval))]

    def apply_scaler(self,scaler):
        self.scaler = scaler
        self.df_train_norm = pd.DataFrame(self.scaler.fit_transform(self.df_train),columns=self.df_train.columns)
        self.df_valid_norm = pd.DataFrame(self.scaler.transform(self.df_valid),columns=self.df_valid.columns)
        self.df_eval_norm  = pd.DataFrame(self.scaler.transform(self.df_eval),columns=self.df_eval.columns)
        

    def plot_loss(self):
        plt.rcParams.update({'font.size': 14})
        plt.figure(figsize=(12,4))
        
        plt.plot(self.hist_loss_train,label = 'train loss')
        plt.plot(self.hist_loss_valid,label = 'valid loss')
        plt.legend()
        plt.grid(True)
        plt.title(self.model._get_name())
        plt.show()
        

    def plot_eval_data(self,cols_names:List[str]=None,max_figures:int=3):
        """
            model is None: plot source data
            model is not None: plot source data and predict data
            cols_names : target name
        """
        assert cols_names is not None
        max_figures = min(max_figures,self.dataloader_eval.__len__())
        for col_name in cols_names:
            for i in range(max_figures):
                input = self.dataloader_eval.dataset[i][0][:,self.index_of_eval_input_column[col_name]].detach()
                target = None
                if col_name in self.targets_columns_names:
                    target = self.dataloader_eval.dataset[i][1][:,self.index_of_eval_target_column[col_name]].detach()
                plt.figure(figsize=(12,4*max_figures))
                plt.subplot(max_figures,1,i+1)
                plt.ylabel(f'{col_name} [normed]')

                plt.plot(self.input_indices,input,label="Inputs",marker='.',zorder = -20)
                if target is not None:
                    plt.scatter(self.target_indices,target,edgecolors='k', label='Targets', c="g", s=64)
                if self.model is not None:
                    input = self.dataloader_eval.dataset[i][0]
                    input = input.view(1,input.shape[0],input.shape[1])
                    pred = self.model(input).flatten()
                    pred = pred.detach()
                    if col_name in self.targets_columns_names:
                        plt.scatter(self.target_indices, pred[self.index_of_eval_target_column[col_name]],
                  marker='X', edgecolors='k', label='Predictions',
                  c='orange', s=64)
                plt.legend()
                plt.title(self.model._get_name())
                plt.show()

    def plot_all_eval_data(self,cols_names:List[str]=None):
        assert cols_names is not None
        
        list_target = []
        list_pred = []
        batch_err_list = []

        for input,target in self.dataloader_eval:
            pred = self.model(input).detach()
            list_target.append(target.reshape(target.shape[1],target.shape[2]))
            list_pred.append(pred)

            err = self.metric(pred,target)
            batch_err_list.append(err)

        self.hist_metric_eval.append(torch.mean(torch.as_tensor(batch_err_list)))

        list_target = torch.stack(list_target,1).view(len(list_target),-1).detach()
        list_pred = torch.stack(list_pred,1).view(len(list_pred),-1).detach()
        
        plt.figure(figsize=(12,6))
        for col_name in cols_names:
            if  col_name not in self.targets_columns_names:
                continue
            print(col_name)
            print(self.index_of_eval_target_column[col_name])
            target = list_target[:,self.index_of_eval_target_column[col_name]].T
            pred = list_pred[:,self.index_of_eval_target_column[col_name]].T
           
            
            plt.plot(target,label="Targets")
            plt.plot(pred,label="Predictions")
            plt.ylabel(f'{col_name} [normed]')
            plt.title(self.model._get_name())
            plt.legend()

            plt.show()

    def train(self,epochs=0, model = None,loss_fn=None,optim=None,metric=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.epochs = epochs
        self.metric = metric

        self.hist_loss_train = []
        self.hist_loss_valid = []
        # self.hist_metric_valid = []
        # self.hist_metric_eval = []

        assert self.model is not None
        assert self.loss_fn is not None
        assert self.optim is not None
        assert self.epochs > 0
        assert self.metric is not None
        p = tqdm(range(self.epochs), total=self.epochs, leave=False)
        for e in p:
            self.train_step()
            self.valid()
            p.set_description("{}/{} epochs | train loss: {} | valid loss: {}".format(e+1,self.epochs,self.hist_loss_train[-1],self.hist_loss_valid[-1]))
        self.plot_loss()

    def train_step(self):
        self.model.train()
        batch_loss_train = []
        for input,target in self.dataloader_train:
            pred = self.model(input)
            loss = self.loss_fn(pred,target)
            batch_loss_train.append(loss.item())

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        self.hist_loss_train.append(torch.mean(torch.as_tensor(batch_loss_train)))

    def valid(self):
        with torch.no_grad():
            self.model.eval()
            batch_loss_valid = []
            batch_err_valid = []
            for input,target in self.dataloader_valid:
                pred = self.model(input)
                loss = self.loss_fn(pred,target)
                err = self.metric(pred,target)
                batch_loss_valid.append(loss.item())
                batch_err_valid.append(err)

        self.hist_loss_valid.append(torch.mean(torch.as_tensor(batch_loss_valid)))
        self.hist_metric_valid.append(torch.mean(torch.as_tensor(batch_err_valid)))

    def eval(self):
        pass

    def heat_map(self):
        matfig = plt.figure(figsize=(32,24))
        font_size = 24

        plt.matshow(self.df.corr(),matfig.number)
        plt.xticks(range(self.df.shape[1]),self.df.columns,fontsize=font_size,rotation=90)
        plt.gca().xaxis.tick_bottom()
        plt.yticks(range(self.df.shape[1]), self.df.columns, fontsize=font_size)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=font_size)
        plt.title("Feature Correlation Heatmap", fontsize=font_size)
        plt.show()

   



