# Overview

This package provides the training code of CBP: Constrained backpropagation based on a pseudo-Lagrange multiplier method. The defining characteristic of the CBP algorithm is the utilization of a Lagrangian function (loss function plus constraint function) as its objective function. We considered various types of constraints — binary, ternary, one-bit shift, and two-bit shift weight constraints. For all cases, the proposed algorithm outperforms the state-of-the-art methods on ImageNet, e.g., 66.6\% and 74.4\% top-1 accuracy for ResNet-18 and ResNet-50 with binary weights, respectively. This highlights CBP as a learning algorithm to address diverse constraints with the minimal performance loss by employing appropriate constraint functions.


### Run environment

+ Python 3.6.9
+ Pytorch 1.7
+ NVIDIA-DALI
    We have used [NVIDIA-DALI](https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html) to implement fast dataloaders.
    You can install it as follows.
```bash
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
```
## Train and evaluation
Download the [ImageNet](https://image-net.org/challenges/LSVRC/2012/) dataset and decompress into the structure like
```bash
    dir/
      data/
          imagenet/
            train/
              n01440764/
                n01440764_10026.JPEG
            ...
            val/
              n01440764/
                ILSVRC2012_val_00000001.JPEG
            ...
```

You can train the model using the following command:
```bash
    python main.py [--model] [--quant] [--batch_size] [--gpu_devices] ...
```

optional arguments:
    --gpu_devices             gpu devices to use (e.g. --gpu_devices 0 1 for using two gpus)

    --model                     model to be quantized (default: resnet18, option: alexnet, resnet18, resnet50)
    --quant                     quantization constraint (default: bin, option: bin, ter, 1bit, 2bit)
    --batch-size                batch size of distilled data (default: 256)
    --lr                           initial learning rate of network parameters (default: 0.001)
    --lr_lambda                 update rate of lagrangian multiplier (default: 0.0001)
    --weight_decay            L2-regularization (default: 0.0001)
    

You should obtain the following results:
| Model      |   Binary  | Ternary  | 1-bit shift | 2-bit shift |
|-------------|:---------:|:----------:|:-----------:|:-----------:|
| AlexNet     |   58.0%  |   58.8%  |    60.8%   |    60.9%   |
| ResNet-18 |   66.6%  |   68.9%  |    69.4%   |    69.4%   |
| ResNet-50 |   74.4%  |   75.0%  |    76.0%   |    75.9%   |


You can evaluate the pre-quantized model using the following command:
```bash
    python evaluate.py [--model] [--quant]
```
optional arguments:
    --model                     quantized model (default: resnet18, option: alexnet, resnet18, resnet50)
    --quant                     quantization constraint (default: bin, option: bin, ter, 1bit, 2bit)

Note that we used initial learning rate of 1e-4 for 1-bit shift and 2-bit shift quantization of AlexNet and ResNet-50.

Disclaimer: Preliminary work. Under review by the 35th Conference on 
Neural Information Processing Systems (NeurIPS 2021). Do not distribute.
