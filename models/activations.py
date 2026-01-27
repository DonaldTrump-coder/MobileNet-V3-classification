import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out

class HSigmoid(nn.Module):
    def __init__(self):
        super(HSigmoid, self).__init__()
    
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out