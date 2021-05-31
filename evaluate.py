import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from resnet_mode import *
from alexnet_mode import *
from module import *
import argparse
from dali import get_imagenet_iter_dali
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url
arch = 'alexnet'
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

############does not use scale factor
parser = argparse.ArgumentParser(description='CBP model evaluation..')
parser.add_argument('--model', default='resnet18', type=str, help='')
parser.add_argument('--quant', default='bin', type=str, help='')
parser.add_argument('--batch_size', default=256, type=int, help='')


args = parser.parse_args()


def main():
    data_dir = os.getcwd()
    if args.model == 'resnet18':
        model=resnet18(mode = args.quant)
    elif args.model == 'resnet50':
        model=resnet50(mode = args.quant)
    elif args.model == 'alexnet':
        model = alexnet(mode = args.quant)

    torch.cuda.set_device(0)
    model.cuda()
    print('model is loaded')
    checkpoint=torch.load(data_dir+"/model/"+args.model+"_"+args.quant+"_prequantized.pth")
    state_dict =checkpoint['model_state_dict']
    state_dict_new = {}
    for p, q in state_dict.items():
        if "module" in p:
            state_dict_new[p[7:]]=q
        else:
            state_dict_new[p]=q

    model_state_dict=model.state_dict()
    model_state_dict.update(state_dict_new)
    model.load_state_dict(model_state_dict)
    valid_loader=get_imagenet_iter_dali(type='val', image_dir='/home/tako/binarize/data/imagenet', batch_size=args.batch_size,num_threads=16, crop=224, device_id= 0 , world_size= 1)

    top1, top5 =validate(model,valid_loader)
    print(top1.item(), top5.item())


def validate(model,data_loader):
    model.eval()
    top1=torch.zeros((1)).cuda()
    top5=torch.zeros((1)).cuda()
    with torch.no_grad():
        for data in data_loader:
            input= data[0]["data"].cuda(non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(non_blocking=True)
            output=model(input)
            prediction=output.data.max(1).indices
            prediction5=output.data.topk(5,dim=1).indices
            top1+=prediction.eq(target.data).sum()
            top5+=prediction5.eq(target.view(-1, 1).expand_as(prediction5)).sum()
        data_loader.reset()
    return top1, top5


if __name__=='__main__':
    main()


