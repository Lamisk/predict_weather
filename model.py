import torch
from torch.functional import Tensor
import torch.nn as nn



class LSTM(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int, output_dim:int, num_layers:int):
        super(LSTM, self).__init__()
        # num_layers: layer in multilayer method, get last hidden layer
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.num_layers = num_layers 
       
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,num_layers=num_layers, batch_first=True) # set batch_first = True :(b,q,data)
        
        self.fc_1 =  nn.Linear(hidden_dim, 256) 
        self.fc_2 = nn.Linear(256, output_dim) 
        

        self.fc = nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
        # Init value
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim) #hidden state
        c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim) #internal cell state
        # Propagate input through LSTM
        output, (last_hidden_state, last_cell_state) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state -> [batch,?,hidden_dim]
        # hn = hn.view(-1, self.hidden_dim) 
        
        # print("hn: ",hn.shape)
        # print("hn[-1]: ",hn[-1].shape)
        # ####
        # out = self.fc(hn[:,-1,:].view(-1,self.hidden_dim))
        out = self.fc(output[:,-1,:])
        # #####
   
        # print("out: ",out.shape)
        return out.view(-1,1,1) #reshape to same size with input -> cal loss and mape

class GRU(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim, num_layers:int, drop_prob:float=0.2):
        # num_layers: layer in multilayer method, get last hidden layer
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Init value
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim)
        out, h = self.gru(x, h_0)
        # Reshaping the outputs to (b, q, h)
        # to fit into the fully connected layer/get last layer
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1,1,1) #reshape to same size with input -> cal loss and mape


# lsmt 1: 4,50 50
# 2: 8,50
# 3: 12,50
# 4:16,50