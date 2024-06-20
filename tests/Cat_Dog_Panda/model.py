import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=0)
        self.pool = nn.MaxPool2d(2,2) 
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(576, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

