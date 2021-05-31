import os
import time
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.hub import load_state_dict_from_url



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
from dali import get_imagenet_iter_dali

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}
parser = argparse.ArgumentParser(description='CBP model training')
parser.add_argument('--model', default='resnet18', type=str, help='')
parser.add_argument('--quant', default='bin', type=str, help='')
parser.add_argument('--lr', default=0.001,type = float, help='')
parser.add_argument('--weight_decay',type = float, default=1e-4, help='')
parser.add_argument('--lr_lambda', default = 1e-4,type = float, help = '')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', type=int, default=256, help='')
parser.add_argument('--period', type=int, default=20, help='')
parser.add_argument('--pretrained', type=bool, default=True, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--rank', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

def main():

    ngpus = torch.cuda.device_count()
    print(args.gpu_devices, len(args.gpu_devices))
    assert len(args.gpu_devices)<= ngpus, 'You chose too many gpus in \'--gpu_devices\''
    args.world_size = len(args.gpu_devices) * args.world_size
    mp.spawn(main_worker, nprocs=len(args.gpu_devices), args=(ngpus, args))
        
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()    
    print("Use GPU: {} for training".format(args.gpu))
    print('weight_decay', args.weight_decay)
        
    args.rank = gpu    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    print('==> Making model..')
    if args.model == 'resnet18':
        model=resnet18(mode = args.quant)
    elif args.model == 'resnet50':
        model=resnet50(mode = args.quant)
    elif args.model == 'alexnet':
        model = alexnet(mode = args.quant)


    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],find_unused_parameters=True)
    group_list = []
    for i in range(args.world_size):
            group_list+=[i]
    group=dist.new_group(group_list)


    data_dir = os.getcwd()
    if args.pretrained ==  True:
        print('==>loading pretrained model..')
        if args.model == 'alexnet':
            print('load alexnet from saved file')
            #path of pretrained alexnet
            path =data_dir+"/model/alexnet_pretrained.pth" 
            checkpoint=torch.load(path)
            model_state_dict=model.state_dict()
            model_state_dict.update(checkpoint['model_state_dict'])
            model.load_state_dict(model_state_dict)

        else:
            print('load resnet from url')
            state_dict = load_state_dict_from_url(model_urls[args.model], progress= True)
            state_dict_new = {}
            for p, q in state_dict.items():
                state_dict_new['module.'+p]=q
            model_state_dict=model.state_dict()
            model_state_dict.update(state_dict_new)
            model.load_state_dict(model_state_dict)

        #set scale factors
        for p in model.modules():
            if isinstance(p,(QConv2d, QLinear)):
                p.scale.data[0] = p.weight.abs().mean()


    print('==> Preparing data..')
    args.batch_size = int(args.batch_size / len(args.gpu_devices))
    valid_loader=get_imagenet_iter_dali(type='val', image_dir=data_dir+'/data/imagenet', batch_size=args.batch_size,num_threads=4, crop=224, device_id=args.gpu, world_size=args.world_size)
    train_loader=get_imagenet_iter_dali(type='train', image_dir=data_dir+'/data/imagenet', batch_size=args.batch_size,num_threads=4, crop=224, device_id=args.gpu, world_size=args.world_size)
    
    '''
    lamb: lagrangian multiplier
    qweight (nqweight): weight to be quantized (not quantized)
    otherparam : parameters except weight
    scale : scale factor of quantization
    factor : factor of each layer. weight quantized to factor * scale
    b : boundary of quantization (median = b * scale)
    param_size : number of elements in qweight
    '''
    lamb, qweight, nqweight, otherparam, factor, b, scale, param_size = getparameters(model)

    #optimizer for network parameters
    optimizer = optim.SGD([{'params':qweight, 'lr':args.lr,'weight_decay':args.weight_decay},{'params':nqweight, 'lr':args.lr,'weight_decay':args.weight_decay},{'params':otherparam, 'lr':args.lr}],momentum=0.9)
    #optimizer for lagrangian multiplier
    optimizer2 = optim.Adam([{'params':lamb, 'lr':args.lr_lambda}])
        
    criterion=nn.CrossEntropyLoss()


    #initialization of unconstrained window
    g = 1
    ucs = 1-1/g

    #inital update of lambda
    updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)

    #save epoch, top1, top5 accuracy and cfs
    progress=np.zeros((1,4))

    #initial epoch, lagsum, period
    epoch_start = 0
    lagsum_pre = 1e10
    period = 0 # lamb and g updated at least 'args.preiod' epoch

    for epoch in range(epoch_start, 1000):
        # train for one epoch
        lagsum = train(model, train_loader, criterion, optimizer, epoch, args, qweight, lamb, scale, factor, b, ucs ,param_size)
        period+=1

        dist.all_reduce(lagsum, group = group)
        print(lagsum, lagsum_pre)
        if  lagsum >= lagsum_pre or period == args.period:
            print('lambda update')

            #update of unconstrained window
            if g<10:
                g+=1
            else:
                g+=10
            ucs =1-1/g

            #update of learning rate
            if g==20:
                adjust_lr(optimizer, 0.1)

            #update of lambda
            updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs)
            
            #reset lagsum and period
            lagsum_pre = 1E10
            period = 0

        else:
            lagsum_pre = lagsum.item()

        #get top1, top5 accuracy
        top1, top5 =validate(model,valid_loader,args)
        dist.all_reduce(top1,group=group)
        dist.all_reduce(top5,group=group)
        print(epoch, top1.item(), top5.item())

        cfs=CFS(qweight,param_size, scale,factor, b)

        #save progress
        progress=np.append(progress,np.array([[epoch, top1.item(), top5.item(), cfs]]),axis=0)
        progress_data=pd.DataFrame(progress)
        progress_data.to_csv(data_dir+"/progress.txt",
        index=False, header=False,sep='\t')

        if epoch%1==0 and args.gpu==0:
            torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'optimizer2_state_dict':optimizer2.state_dict(),
                    'epoch':epoch,
                    'lamb':lamb,
                    'lagsum_pre':lagsum_pre,
                    'period':period,
                    'ucs':ucs,
                    'g':g,
                    'progress':progress,
                    },data_dir+"/model/"+args.model+"_"+args.quant+"_%d.pth"%(epoch))

def train(model, train_loader, criterion, optimizer, epoch, args, qweight, lamb, scale, factor, b, ucs,param_size):
    lagsum=torch.zeros((1)).cuda(args.gpu)
    model.train()
    idx = 0
    y = time.time()
    for data in (train_loader):
        # measure data loading time
        if idx%1000==1:
            print(args.gpu,idx,time.time()-y,loss_network.item(),  lag.item(),CFS(qweight,param_size, scale,factor, b))
            y=time.time()
        idx+=1
        input = data[0]["data"].cuda(args.gpu, non_blocking=True)
        target = data[0]["label"].squeeze().long().cuda(args.gpu, non_blocking=True)
        output = model(input)

        loss_network=criterion(output,target)
        const=torch.zeros(1).cuda()
        for i in range(len(lamb)):
            const = const+constraints(qweight[i],lamb[i].detach(),scale[i],factor[i],b[i], ucs)
        lag = loss_network + const
        lagsum += lag.detach()
        optimizer.zero_grad()
        lag.backward(retain_graph=True)
        torch.nn.utils.clip_grad_value_(parameters=model.parameters(),clip_value=1)
        optimizer.step()
        for p in qweight:
            p.data.clamp_(min = -1, max = 1)
    train_loader.reset()

    return lagsum

def validate(model,data_loader,args):
    model.eval()
    top1=torch.zeros((1)).cuda(args.gpu)
    top5=torch.zeros((1)).cuda(args.gpu)
    with torch.no_grad():
        for data in data_loader:
            input= data[0]["data"].cuda(args.gpu,non_blocking=True)
            target = data[0]["label"].squeeze().long().cuda(args.gpu,non_blocking=True)
            output=model(input)
            prediction=output.data.max(1).indices
            prediction5=output.data.topk(5,dim=1).indices
            top1+=prediction.eq(target.data).sum()
            top5+=prediction5.eq(target.view(-1, 1).expand_as(prediction5)).sum()
        data_loader.reset()
    return top1, top5
def getparameters(model):
    lamb=[] #Lagrangian multiplier
    qweight=[] #weight to be quantized
    nqweight=[] #weight not to be quantized
    otherparam=[] #other parameters such as batchnorm , bias,...
    factor=[] #factors of each quantized layers
    b = [] #b of each quantized leayers
    scale = [] #scale factor of each quantized layers
    param_size = 0
    for p in model.modules():
        if isinstance(p,(QConv2d, QLinear)):
            qweight+=[p.weight]
            lamb+=[Variable(torch.full(p.weight.shape,0).float().cuda(),requires_grad=True)]
            if p.bias!=None:
                otherparam+=[p.bias]
            scale+=[p.scale]
            factor+=[p.factor]
            b += [p.b]
            param_size+=p.weight.numel()

        elif isinstance(p,(nn.Conv2d, nn.Linear)):
            nqweight+=[p.weight]
            if p.bias!=None:
                otherparam+=[p.bias]
        elif isinstance(p,(nn.BatchNorm2d,nn.BatchNorm1d)):
            otherparam+=[p.weight]
            otherparam+=[p.bias]
    return lamb, qweight, nqweight, otherparam, factor, b, scale, param_size

def updatelambda(optimizer2, qweight, lamb, scale, factor, b, ucs):
    const=torch.zeros(1).cuda()
    for i in range(len(lamb)):
        const = const+constraints(qweight[i].detach(),lamb[i],scale[i],factor[i],b[i], ucs)
    optimizer2.zero_grad()
    (-const).backward(retain_graph=True)
    optimizer2.step()


def adjust_lr(optimizer, decrease_rate):
    for p in optimizer.param_groups:
        p['lr']*=decrease_rate


def constraints(weight,lamb,scale, factor, b, ucs):
    out = constraint().apply(weight,scale,factor,b,ucs)
    return (out*lamb).sum()

def CFS(weight,size, scale,factor, b):
    cfstotal=0
    for p, q, r,s in zip(weight,scale,factor, b):
        cfs = constraint().apply(p,q,r,s,1)
        cfstotal+=cfs.sum()
    return cfstotal.item()/size

if __name__=='__main__':
    main()


