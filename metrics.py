import torch
def MAPE(pred, target):
    target, pred = torch.as_tensor(target), torch.as_tensor(pred)
    return torch.mean(torch.abs((target - pred) / target)) * 100