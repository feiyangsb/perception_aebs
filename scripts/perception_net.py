import torch.nn as nn
import torch.nn.functional as F

class PerceptionNet(nn.Module):
    def __init__(self):
        super(PerceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2, padding=0, bias=False)
        #nn.init.xavier_uniform(self.conv1.weight)
        self.bn2d1 = nn.BatchNorm2d(24, eps=1e-03, affine=False)

        self.conv2 = nn.Conv2d(24, 36, 5, stride=2, padding=0, bias=False)
        #nn.init.xavier_uniform(self.conv2.weight)
        self.bn2d2 = nn.BatchNorm2d(36, eps=1e-03, affine=False)

        self.conv3 = nn.Conv2d(36, 48, 5, stride=2, padding=0, bias=False)
        #nn.init.xavier_uniform(self.conv3.weight)
        self.bn2d3 = nn.BatchNorm2d(48, eps=1e-03, affine=False)

        self.conv4 = nn.Conv2d(48, 64, 3, padding=0, bias=False)
        #nn.init.xavier_uniform(self.conv4.weight)
        self.bn2d4 = nn.BatchNorm2d(64, eps=1e-03, affine=False)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=0, bias=False)
        #nn.init.xavier_uniform(self.conv5.weight)
        self.bn2d5 = nn.BatchNorm2d(64, eps=1e-03, affine=False)

        self.dropout = nn.Dropout(0.3)

        #self.fc1 = nn.Linear(65536, 100)
        self.fc1 = nn.Linear(28224, 100)
        #nn.init.xavier_uniform(self.fc1.weight)
        self.fc2 = nn.Linear(100, 50)
        #nn.init.xavier_uniform(self.fc2.weight)
        self.fc3 = nn.Linear(50, 10)
        #nn.init.xavier_uniform(self.fc3.weight)
        self.fc4 = nn.Linear(10, 1)
        #nn.init.xavier_uniform(self.fc4.weight)
    
    def forward(self, x):
        x = self.conv1(x)
        #x = F.relu(x)
        x = F.relu(self.bn2d1(x))
        x = self.conv2(x)
        #x = F.relu(x)
        x = F.relu(self.bn2d2(x))
        x = self.conv3(x)
        #x = F.relu(x)
        x = F.relu(self.bn2d3(x))
        x = self.conv4(x)
        #x = F.relu(x)
        x = F.relu(self.bn2d4(x))
        x = self.conv5(x)
        #x = F.relu(x)
        x = F.relu(self.bn2d5(x))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        
        return x