import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deepsets_invariant import DeepSetsInvariant

class DigitSumImagePhi(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.dropout_conv1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout_conv2 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(320, 50)
        self.dropout_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(50,10)
        self.dropout_fc2 = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout_conv1(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout_conv2(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout_fc2(x)
        return x


class DigitSumImageRho(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10,10)
        #self.dropout_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(10,1)
        #self.dropout_fc2 = nn.Dropout()
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = self.dropout_fc2(x)
        return x

def digitsum_image50():
    phi = DigitSumImagePhi()
    rho = DigitSumImageRho()
    model = DeepSetsInvariant(phi, rho)
    return model

def digitsum_image(model_name):
    return{
        'digitsum_image50': digitsum_image50(),
    }[model_name]