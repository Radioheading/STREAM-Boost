import torch
from torch import dropout
import torch.nn as nn
import torch.nn.functional as F
from utils import ReverseLayerF


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
            

            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, 3 * 28 * 28)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
    
    def sureness(self, x):
        x = x.view(x.size(0), -1)  # 将输入张量重新reshape成适合线性层的形状
        output = self.classifier(x)
        # print(normalized_output)
        normalized_output = F.softmax(output, dim=1)
        # print(normalized_output)
        sure = torch.max(normalized_output, dim=1).values
        # print(sure)
        # return the percentage of data with sureness < 0.5
        return sure, sum(sure < 0.5)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=3 * 28 * 28, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x
