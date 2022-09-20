import torch
# import numpy as np 
import torch.nn as nn
import torch

def MAPE(x, y):
    return torch.mean(torch.abs((y - x) / y)) 

def RMSE(x, y):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(x, y))

def R2(x, y):
    target_mean = torch.mean(y)
    ss_tot = torch.sum((y - target_mean) ** 2)
    ss_res = torch.sum((y - x) ** 2)
    
    r2 = 1 - ss_res / ss_tot
    
    return r2

def indicator(x, y):    
    mape = MAPE(x, y).item()
    rmse = RMSE(x, y).item()
    r2 = R2(x, y).item()
    return {'MAPE': mape, 'RMSE': rmse, 'R2': r2}
    