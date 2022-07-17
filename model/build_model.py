import torch
from torch import nn
import torchvision
from model.sa_resnet import sa_resnet50

from model.pysa_resnet import pysaresnet50
from model.py_resnet import pyconvresnet50

class ResNet50(nn.Module):

    def __init__(self, N_LABELS, isTrained):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x

class SA_ResNet50(nn.Module):

    def __init__(self, N_LABELS, isTrained):
        super(SA_ResNet50, self).__init__()
        self.sa_resnet50 = sa_resnet50(pretrained=isTrained)
        num_ftrs = self.sa_resnet50.fc.in_features
        self.sa_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    def forward(self, x):
        x = self.sa_resnet50(x)
        return x

class PY_ResNet50(nn.Module):

    def __init__(self, N_LABELS, isTrained):
        super(PY_ResNet50, self).__init__()
        self.py_resnet50 = pyconvresnet50(pretrained=isTrained)
        num_ftrs = self.py_resnet50.fc.in_features
        self.py_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    def forward(self, x):
        x = self.py_resnet50(x)
        return x

class PYSA_ResNet50(nn.Module):

    def __init__(self, N_LABELS, isTrained):
        super(PYSA_ResNet50, self).__init__()
        self.pysa_resnet50 = pysaresnet50(pretrained=isTrained)
        num_ftrs = self.pysa_resnet50.fc.in_features
        self.pysa_resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    def forward(self, x):
        x = self.pysa_resnet50(x)
        return x




class DenseNet121(nn.Module):

    def __init__(self, N_LABELS, isTrained):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x