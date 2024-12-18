from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder
from PIL import Image
from torchvision.transforms import v2
import torch
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
import random

MEAN = (0.7235, 0.6475, 0.7527)
STD = (0.3160, 0.3376, 0.3030)

def get_mhist_loader(root, transform, split='train', mixup=False, cutmix=False, dist=False, **kwargs):
    dataset = ImageFolder(root,
                        transform=transform,
                        loader=Image.open,
                        is_valid_file=lambda x : split in x,
                        allow_empty=False)
    print('[DATASET INIT]', 'dataset_size:',len(dataset))
    print('[DATASET INIT]', 'dataset classes:', dataset.class_to_idx)
    cutmix_f = v2.CutMix(num_classes=len(dataset.classes))
    mixup_f = v2.MixUp(num_classes=len(dataset.classes))
    cutmix_or_mixup = v2.RandomChoice([cutmix_f, mixup_f])
    if mixup and cutmix:
        collate_fn = lambda batch : cutmix_or_mixup(*default_collate(batch)) if random.random() < 0.2 else default_collate(batch)
    elif mixup:
        collate_fn = lambda batch : mixup_f(*default_collate(batch)) if random.random() < 0.2 else default_collate(batch)
    elif cutmix:
        collate_fn = lambda batch : cutmix_f(*default_collate(batch)) if random.random() < 0.2 else default_collate(batch)
    else:
        collate_fn = None
    labels = np.array([_[1] for _ in dataset])
    return DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset) if dist else None,
        **kwargs
    ), labels

def get_preproc_transform(input_size=(224,224),
                          degrees=45,
                          translate=0.06,
                          shear=0.9,
                          scale=0.2,
                          hue=0.1,
                          saturation=0.3,
                          brightness=0.3,
                          contrast=0.3,
                          sharpness=1.3,
                          norm=True):
    return v2.Compose([
        v2.PILToTensor(),
        # shape
        v2.RandomAffine(degrees=degrees, translate=(-translate, translate), shear=(-shear,shear,-shear,shear)),
        v2.RandomResizedCrop(size=input_size, scale=(1-scale, 1+scale), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        # color
        v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue), # random hed
        v2.RandomAdjustSharpness(sharpness_factor=sharpness, p=0.3),
        # v2.RandomApply([v2.GaussianBlur(3)], p=0.3), # use random blur
        # normalize
        v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
        v2.Normalize(mean=MEAN, std=STD) if norm else \
        v2.Normalize(mean=(0,0,0), std=(1,1,1)),  # typically from ImageNet
    ])

def plot_train_images(images, targets, fname='images.jpg', max_size=640, max_subplots=16):
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    if os.path.isfile(fname):  # do not overwrite
        return None

    assert isinstance(images, torch.Tensor) and isinstance(targets, torch.Tensor)
    images = images.cpu().float().numpy()
    images = (images  * np.array(STD)[:, None, None] + np.array(MEAN)[:, None, None]) * 255
    targets = targets.cpu().numpy()


    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        # # Draw target labels
        # for target in targets:
        #     if len(target) > 1:
        #         label = ''
        #         for i, p in enumerate(target.tolist()):
        #             label += f'{i}: {p:.2f} '
        #     else:
        #         label = f'{target.item()}'
        #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 8, thickness=tf)[0]
        #     cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 8, [220, 220, 220], thickness=tf,
        #             lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def get_test_transform(input_size=(224,224), half=False, norm=True):
    return v2.Compose([
        v2.PILToTensor(),
        v2.Resize(input_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD) if norm else \
        v2.Normalize(mean=(0,0,0), std=(1,1,1)),
        v2.ToDtype(torch.float16 if half else torch.float32, scale=False),
    ])

def calculate_mean_std(root, transform, split='train', mixup=False, cutmix=False, epochs=5, **kwargs):
    dataset = ImageFolder(root,
                        transform=transform,
                        loader=Image.open,
                        is_valid_file=lambda x : split in x,
                        allow_empty=False)
    print('[DATASET INIT]', 'dataset_size:',len(dataset))
    print('[DATASET INIT]', 'dataset classes:', dataset.class_to_idx)
    cutmix = v2.CutMix(num_classes=len(dataset.classes))
    mixup = v2.MixUp(num_classes=len(dataset.classes))
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    if mixup and cutmix:
        collate_fn = lambda batch : cutmix_or_mixup(*default_collate(batch))
    elif mixup:
        collate_fn = lambda batch : mixup(*default_collate(batch))
    elif cutmix:
        collate_fn = lambda batch : cutmix(*default_collate(batch))
    else:
        collate_fn = None
    loader =  DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        **kwargs
    )
    avg = torch.zeros(3)
    total_bs = 0
    for i in range(epochs):
        for imgs, _ in loader:
            bs = imgs.shape[0]
            avg = avg * (total_bs / (total_bs+bs)) + imgs.transpose(0,1).flatten(1,3).mean(1) * (bs / (total_bs+bs))
            total_bs += bs
        print('[DATASET STAT]', 'epoch', i, 'mean: ', avg)
    var = torch.zeros(3)
    total_bs = 0
    for i in range(epochs):
        for imgs, _ in loader:
            bs = imgs.shape[0]
            var = var * (total_bs / (total_bs+bs)) + ((imgs.transpose(0,1).flatten(1,3) - avg.reshape(3,1)) ** 2).mean(1) * (bs / (total_bs+bs))
            total_bs += bs
        print('[DATASET STAT]', 'epoch', i, 'std: ', var.sqrt())

    return

if __name__ == "__main__":
    # loader = get_mhist_loader('dataset/mhist/images', get_preproc_transform(), split='train', mixup=True, cutmix=True, 
    #                         batch_size=2, 
    #                         shuffle=True,
    #                         num_workers=1,
    #                         pin_memory=True,
    #                         prefetch_factor=4)
    # for img, label in loader:
    #     print(img.shape)
    #     print(label)
    #     break

    calculate_mean_std('dataset/mhist/images', get_test_transform(norm=False), split='test', 
                       epochs=5,
                       mixup=True, cutmix=True, 
                       batch_size=64, 
                       shuffle=True)