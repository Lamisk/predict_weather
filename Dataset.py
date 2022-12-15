#%%
from torch.functional import Tensor
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self,input:Tensor,seq_len:int,out_width:int=2) -> None:
        super().__init__()
        self.input = input
        self.seq_len = seq_len
        self.out_width = out_width

    def __getitem__(self, index):
        input  = self.input[index:index+self.seq_len]
        target = self.input[index+self.seq_len:index+self.seq_len+self.out_width] # Target la mang nhieu chieu, gom 2 phan tu ke tiep: FAIL - infinity loop
        return input,target

    def __len__(self):
        return self.input.shape[0] - self.seq_len - self.out_width -1



#%%
import torch

input = torch.randn(7,5)
seq_len = 4
print(input)
print('*********************')
ds = Dataset(input=input,seq_len=seq_len,out_width=2)
for idx,sample in enumerate(ds):
    input = sample[0]
    target = sample[1]
    print(input)
    print(target)
    print('----------------------------')
    if idx==10:
        break
# %%
