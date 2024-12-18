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

博客

- 边界框、锚框、预测框 https://blog.csdn.net/lzzzzzzm/article/details/120621582
- 遗传算法和kmeans聚类生成锚框

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

        https://blog.csdn.net/dgvv4/article/details/123988282

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
tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # run directory
# class frequency
if tb_writer:
    tb_writer.add_histogram('classes', c, 0)
# Plot
if ni < 3:
    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
    result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
    if tb_writer and result is not None:
        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
        # tb_writer.add_graph(model, imgs)  # add model to tensorboard
# metrics
if tb_writer:
    tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
            'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
            'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
    for x, tag in zip(list(mloss[:-1]) + list(results), tags):
        tb_writer.add_scalar(tag, x, epoch)
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
    --cfg models/yolov4m-mish.yaml \
    --data data/neu.yaml \
    --epochs 500 \
    --batch-size 16 \
    --img-size 200 200 \
    --name yolov4m \
    --device 0 \
    #--noautoanchor \
    #--cache-images
    #--hyp
    #--resume
    #--nosave
    #--notest
    #--multi-scale
    #--single-cls
```

## utils/utils.py

### plot_images

输出0.3以上置信度的框，且如果文件存在，不重新绘图

### build_targets

- https://blog.csdn.net/u013066730/article/details/126969286
- https://github.com/ultralytics/yolov5/issues/471
- https://zhuanlan.zhihu.com/p/399153002
- https://blog.csdn.net/wxd1233/article/details/126148680
- 正样本分配 https://blog.csdn.net/u013066730/article/details/126969286

## loss

yolo loss解读: https://zhuanlan.zhihu.com/p/591833099

```python
# compute_loss
# Define criteria
lcls, lbox, lobj = ft([0]).to(device), ft([0]).to(device), ft([0]).to(device)
tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets

BCEcls = nn.BCEWithLogitsLoss(pos_weight=ft([h['cls_pw']]), reduction=red).to(device)
BCEobj = nn.BCEWithLogitsLoss(pos_weight=ft([h['obj_pw']]), reduction=red).to(device)

# focal loss
g = h['fl_gamma']  # focal loss gamma
if g > 0:
    BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

# per output
    nt = 0  # number of targets
    np = len(p)  # layers of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0]).to(device)  # target obj

        nb = b.shape[0]  # number of targets
        if nb:
            nt += nb  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # lbox

            # tObj

            # lcls

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h['giou'] * s
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    lcls *= h['cls'] * s
    bs = tobj.shape[0]  # batch size
    if red == 'sum':
        g = 3.0  # loss gain
        lobj *= g / bs
        if nt:
            lcls *= g / nt / model.nc
            lbox *= g / nt

    loss = lbox + lobj + lcls
```

### cls_loss

```python
if model.nc > 1:  # cls loss (only if multiple classes)
    t = torch.full_like(ps[:, 5:], cn).to(device)  # targets
    t[range(nb), tcls[i]] = cp
    lcls += BCEcls(ps[:, 5:], t)  # BCE
s = 3 / np  # output count scaling
lcls *= h['cls'] * s
if red == 'sum':
    g = 3.0  # loss gain
    if nt:
        lcls *= g / nt / model.nc
```

### giou_loss/box_loss

```python
# GIoU
pxy = ps[:, :2].sigmoid() * 2. - 0.5
pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
giou = bbox_iou(pbox.t(), tbox[i], x1y1x2y2=False, GIoU=True)  # giou(prediction, target)
lbox += (1.0 - giou).sum() if red == 'sum' else (1.0 - giou).mean()  # giou loss
s = 3 / np  # output count scaling
lbox *= h['giou'] * s
if red == 'sum':
    g = 3.0  # loss gain
    if nt:
        lbox *= g / nt
```

### obj_loss

```python
# Obj
tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio
lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
s = 3 / np
lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
if red == 'sum':
    bs = tobj.shape[0]  # batch size
    g = 3.0  # loss gain
    lobj *= g / bs
```

## test.py - metrics

```python
# Append statistics (correct, conf, pcls, tcls)
stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

p, r, ap, f1, ap_class = ap_per_class(*stats)
p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
```

### mAP_{0.5}

0.5的以上iou作为输出预测框，有tp，fp，fn，没有tn

- precision tp / tp + fp
- recall = tp / tp + fn
- ap的计算，多少的objectiveness/conf以上作为tp

### mAP_{0.5:0.95}

0.5-0.95，10各iou，各自为基准确定输出预测框，最后10个值取平均

### precision

iou取的是0.5时的precision, positive_thresh=0.1

### recall

iou取的是0.5时的recall, positive_thresh=0.1

## models

### parse_model

1. parse_anchors

    ```python
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ```

2. parse backbone and head

    ```python
    # [from, number, module, args]
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):

    # block: scale depth
    n = max(round(n * gd), 1) if n > 1 else n  # depth gain

    # conv ro conv blocks: process cin and cout
    if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:

    # cin and cout
    c1, c2 = ch[f], args[0]

    # scale channel and multiple of 8
    c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

    # conv blocks: append depth to args
    if m in [BottleneckCSP, BottleneckCSP2, SPPCSP, VoVCSP, C3]:

    # batch_norm2d: input channel
    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    # concat: write this layer output channel
    elif m is Concat:
        c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
    # detect: add all input channels, number of anchors
    elif m is Detect:
        args.append([ch[x + 1] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)
    # else: write this layer output channel
    else:
        c2 = ch[f]
    ```