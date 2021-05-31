import torch
import torch.nn as nn
from module import *
from torch.autograd import Variable

def conv(in_planes, out_planes, kernel_size, stride=1,padding = 1,  groups=1, dilation=1, mode=None,bias = False):

    if mode in ['bin', 'ter', '1bit', '2bit']:
        return QConv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation, mode = mode)
    else:
        print('entered mode not exist. conventional convolution will be set')
        return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias=bias, dilation=dilation)

def linear(in_features, out_features, bias = False, mode = None):
    if mode in ['bin', 'ter', '1bit', '2bit']:
        return QLinear(in_features, out_features, bias, mode = mode)

    else:
        print('entered mode not exist, conventional linear function will be set')
        return nn.Linear(in_features, out_features, bias)


class alexnet(nn.Module):

    def __init__(self, num_classes=1000, mode = 'Q'):
        super(alexnet, self).__init__()
        self.features = nn.Sequential(
            conv(3, 64, kernel_size=11, stride=4, padding=2,bias=False,mode = None),
            nn.BatchNorm2d(64,eps=1e-4,momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(64, 192, kernel_size=5, padding=2,bias=False,mode = mode),
            nn.BatchNorm2d(192,eps=1e-4,momentum=0.1),           

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            conv(192, 384, kernel_size=3, padding=1,bias=False,mode = mode),
            nn.BatchNorm2d(384,eps=1e-4,momentum=0.1),           

            nn.ReLU(),
            conv(384, 256, kernel_size=3, padding=1,bias=False,mode = mode),
            nn.BatchNorm2d(256,eps=1e-4,momentum=0.1),           

            nn.ReLU(),
            conv(256, 256, kernel_size=3, padding=1,bias=False,mode = mode),
            nn.BatchNorm2d(256,eps=1e-4,momentum=0.1),           

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            linear(256 * 6 * 6, 4096,bias=False,mode = mode),
            nn.BatchNorm1d(4096,eps=1e-4, momentum=0.1),

            nn.ReLU(),
            linear(4096, 4096,bias=False,mode = mode),
            nn.BatchNorm1d(4096,eps=1e-4, momentum=0.1),

            nn.ReLU(),
            linear(4096, 1000,bias=True,mode = None),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d,nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


