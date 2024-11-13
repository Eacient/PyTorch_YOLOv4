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

### random_perspective/random_affine()

```python
img4, labels4 = random_affine(img4, labels4,
                              degrees=self.hyp['degrees'],
                              translate=self.hyp['translate'],
                              scale=self.hyp['scale'],
                              shear=self.hyp['shear'],
                              border=self.mosaic_border)  # border to remove
img, labels = random_affine(img, labels,
                            degrees=hyp['degrees'],
                            translate=hyp['translate'],
                            scale=hyp['scale'],
                            shear=hyp['shear'])
```

### random_hsv

```python
augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
```

## train.py

### hyp

```python
'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
'momentum': 0.937,  # SGD momentum/Adam beta1
'weight_decay': 5e-4,  # optimizer weight decay
'giou': 0.05,  # giou loss gain
'cls': 0.5,  # cls loss gain
'cls_pw': 1.0,  # cls BCELoss positive_weight
'obj': 1.0,  # obj loss gain (*=img_size/320 if img_size != 320)
'obj_pw': 1.0,  # obj BCELoss positive_weight
'iou_t': 0.20,  # iou training threshold
'anchor_t': 4.0,  # anchor-multiple threshold
'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
```

### tb_writer

```python
log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # run directory
```

### opt

#### opt codes
```python
opt = parser.parse_args()
epochs, batch_size, total_batch_size, weights, rank = \
    opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.local_rank
with open(opt.data) as f:
    data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
# Create model
model = Model(opt.cfg, nc=nc).to(device)
# Image sizes
gs = int(max(model.stride))  # grid size (max stride)
imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
# SyncBatchNorm
if opt.sync_bn and device.type != 'cpu' and rank != -1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    print('Using SyncBatchNorm()')
# Trainloader
dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt, hyp=hyp, augment=True,
                                        cache=opt.cache_images, rect=opt.rect, local_rank=rank,
                                        world_size=opt.world_size)
# Testloader
if rank in [-1, 0]:
    # local_rank is set to -1. Because only the first process is expected to do evaluation.
    testloader = create_dataloader(test_path, imgsz_test, total_batch_size, gs, opt, hyp=hyp, augment=False,
                                   cache=opt.cache_images, rect=True, local_rank=-1, world_size=opt.world_size)[0]
# Check anchors
if not opt.noautoanchor:
    check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
# Multi-scale
if opt.multi_scale:
    sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
    sf = sz / max(imgs.shape[2:])  # scale factor
    if sf != 1:
        ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
        imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
# Loss
loss, loss_items = compute_loss(pred, targets.to(device), model)  # scaled by batch_size
if rank != -1:
    loss *= opt.world_size  # gradient averaged between devices in DDP mode
# Calculate mAP
if not opt.notest or final_epoch:  
    results, maps, times = test.test(opt.data,
                                     batch_size=total_batch_size,
                                     imgsz=imgsz_test,
                                     save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                     model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                     single_cls=opt.single_cls,
                                     dataloader=testloader,
                                     save_dir=log_dir)
# Write
with open(results_file, 'a') as f:
    f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
if len(opt.name) and opt.bucket:
    os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))
# Save model
save = (not opt.nosave) or (final_epoch and not opt.evolve)
if save:
    with open(results_file, 'r') as f:  # create checkpoint
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': f.read(),
                'model': ema.ema.module if hasattr(ema, 'module') else ema.ema,
                'optimizer': None if final_epoch else optimizer.state_dict()}
#strip optimizer
ispt = f2.endswith('.pt')  # is *.pt
strip_optimizer(f2) if ispt else None  # strip optimizer
os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload
# Finish
if not opt.evolve:
    plot_results(save_dir=log_dir)  # save as results.png
```

#### opt cmd

```bash
python train.py \
    --cfg models/yolov4s-mish.yaml \
    --data data/neu.yaml \
    --epochs 300 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4s \
    --device cpu \
    #--noautoanchor \
    #--cache-images
    #--hyp
    #--resume
    #--nosave
    #--notest
    #--multi-scale
    #--single-cls
```