from typing import Dict, List
import matplotlib.pyplot as plt
import torch

def plot_performance(dict_models:dict,col_names:List[str],metric_valid:List,metric_eval:List,metric):
    for col_name in col_names:
        plt.figure(figsize = (12,8))
        metric_name = metric.__name__
        width = 0.3
        err_valid_list = metric_valid
        err_eval_list = metric_eval
        print(err_valid_list)
        print(err_eval_list)
        x = torch.arange(len(dict_models))
        plt.ylabel(f'{metric_name} [{col_name}] [normalized]')
        plt.bar(x - 0.18, err_valid_list, width, label='Validation')
        plt.bar(x + 0.18, err_eval_list, width, label='Eval')
        plt.grid(True)
        plt.xticks(ticks=x, labels=dict_models.keys(), rotation=45)
        _ = plt.legend()
       
        plt.show()