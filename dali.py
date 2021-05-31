import time
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from PIL import Image
import numpy as np
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False,world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=world_size, shuffle_after_epoch=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, interp_type=types.INTERP_TRIANGULAR)
        #self.res = ops.Resize(device = "gpu", resize_shorter = 256)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            crop = [224,224],
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.colorjitter=ops.ColorTwist(device = "gpu")
        self.coin = ops.CoinFlip(probability=0.5)
        self.rng1 = ops.Uniform(range=[0.8, 1.2])
        self.rng2 = ops.Uniform(range=[-0.1, 0.1])
        self.rng3 = ops.Uniform(range=[0,1])

        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        saturation = self.rng1()
        contrast = self.rng1()
        brightness = self.rng1()
        hue = self.rng2()
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.colorjitter(images.gpu(), saturation=saturation, contrast=contrast, brightness=brightness, hue=hue)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=device_id, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.DALIImageType.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, crop, val_size=256,
                           world_size=1):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir=image_dir + '/train',
                                    crop=crop, world_size=world_size)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size, world_size=world_size)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size)
        return dali_iter_val

