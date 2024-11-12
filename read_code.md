# read_code

## 相关论文

pan: neck的设计，路径融合网络

https://blog.csdn.net/flyfish1986/article/details/110520667

cspnet: 

https://zhuanlan.zhihu.com/p/393778545

yolov4: 架构解读

https://blog.csdn.net/m0_57787115/article/details/130588237

yolov4: 论文解读

https://zhuanlan.zhihu.com/p/137393450

## utils/datasets.py

### rect

choose rect size rather than fixed size of the batch

### cache_images

keep img in memory in dict format

### augmentation and hyp

1. normal load-img

    load_img()

2. augment image space

    random_perspective()

3. augment color space

    augment_hsv()

4. apply cutouts (not used)
5. flip up-down
6. flip left-right

### mosaic augmentation

is default augmentation when not rect

1. mosaic load

    1. 4 mosaic load
    2. random_perspective()

3. mix-up load
4. augment color space
5. apply cutouts (not used)
6. flip up-down
7. flip left-right