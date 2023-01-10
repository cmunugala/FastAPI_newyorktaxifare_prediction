import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class NeuralNetwork(nn.Module):
    def __init__(self,n_input,n_hidden1,n_hidden2,n_hidden3,n_out):
        super(NeuralNetwork,self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input,n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1,n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2,n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3,n_out)
        )
    
    def forward(self,x):
        outputs = self.linear_relu_stack(x)
        return outputs

