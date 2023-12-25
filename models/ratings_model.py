import torch
import torchvision.models as models
import torch.nn as nn


class RatingsTestModel(nn.Module):
    def __init__(self, n_classes):
        super(RatingsModel, self).__init__()
        self.fc1 = nn.Linear(6040, 3020)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(3020, 1510)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(1510, 755)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(755, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.fc5 = nn.Linear(512, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.2)

        self.fc6 = nn.Linear(256, n_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        x = self.relu5(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = self.sigmoid(x)
        return x

class RatingsExtractModel(nn.Module):
    def __init__(self):
        super(RatingsExtractModel, self).__init__()
        self.fc1 = nn.Linear(6040, 3020)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)

        self.fc2 = nn.Linear(3020, 1510)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

        self.fc3 = nn.Linear(1510, 755)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)

        self.fc4 = nn.Linear(755, 512)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.2)
        
        self.fc5 = nn.Linear(512, 256)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout4(x)

        x = self.fc5(x)
        return x