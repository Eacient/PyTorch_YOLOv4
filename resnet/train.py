import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch import autocast, GradScaler

import sys
sys.path.append('.')
import resnet.test as test  # import test.py to get mAP after each epoch
from resnet.models.resnet import Resnet, get_opt_param_groups_cnn, get_warmup_tuner
from resnet.utils.datasets import get_mhist_loader, get_preproc_transform, get_test_transform, plot_train_images 
from utils import google_utils
from utils.datasets import *
from utils.utils import *

mixed_precision=False
half_test=False
# Hyperparameters
hyp = {'optimizer': 'SGD',  # ['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0,  # image HSV-Value augmentation (fraction)
       'contrast': 0,
       'sharpness': 1,
       'degrees': 0,  # image rotation (+/- deg)
       'translate': 0,  # image translation (+/- fraction)
       'shear': 0, # image shear (+/- deg)
       'scale': 0.2,  # image scale (+/- gain)
       'mixup': False,
       'cutmix': False,}  


def loss_func(pred, target):
    return nn.functional.cross_entropy(pred, target)
# loss_func = FocalLoss(nn.BCEWithLogitsLoss())

def get_optimizer(model, total_batch_size, hyp):
    # Optimizer
    nbs = 64  # nominal batch size
    # default DDP implementation is slow for accumulation according to: https://pytorch.org/docs/stable/notes/ddp.html
    # all-reduce operation is carried out during loss.backward().
    # Thus, there would be redundant all-reduce communications in a accumulation procedure,
    # which means, the result is still right but the training speed gets slower.
    # TODO: If acceleration is needed, there is an implementation of allreduce_post_accumulation
    # in https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/run_pretraining.py
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    pg = get_opt_param_groups_cnn(model, wd=hyp['weight_decay'])

    if hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        optimizer = optim.Adam(pg, lr=hyp['lr0'], betas=(hyp['momentum'], 0.95))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=False)

    del pg

    return optimizer, accumulate, nbs

def get_dataloader(hyp, train_path, test_path, batch_size, total_batch_size, opt_sz, rank=-1, tb_writer=None):
    # Image sizes
    gs = 32  # grid size (max stride)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt_sz]  # verify imgsz are gs-multiples
    # Trainloader
    pre_transforms = get_preproc_transform((imgsz, imgsz), hyp['degrees'], hyp['translate'], hyp['shear'], hyp['scale'], 
                                           hyp['hsv_h'], hyp['hsv_s'], hyp['hsv_v'], hyp['contrast'], hyp['sharpness'])
    dataloader, labels_stat = get_mhist_loader(train_path, pre_transforms, split='train', dist=(rank != -1),
                                  mixup=hyp['mixup'], cutmix=hyp['cutmix'], 
                                  batch_size=batch_size,
                                  num_workers=min([os.cpu_count() // opt.world_size, batch_size if batch_size > 1 else 0, 8]),
                                  shuffle=True,
                                  pin_memory=True,
                                  prefetch_factor=2)

    # Testloader
    if rank in [-1, 0]:
        # local_rank is set to -1. Because only the first process is expected to do evaluation.
        test_transform = get_test_transform((imgsz_test, imgsz_test), half = (half_test and device.type != 'cpu'))
        testloader, _ = get_mhist_loader(test_path, test_transform, split='test', dist=False,
                                      batch_size=total_batch_size,
                                      num_workers=min([os.cpu_count() // opt.world_size, batch_size if batch_size > 1 else 0, 8]),
                                      shuffle=True,
                                      pin_memory=True,
                                      prefetch_factor=2)
    else:
        testloader = None
    if rank in [0, -1]:
        print('[DATALOADER INIT]', 'Image sizes %g train, %g test' % (imgsz, imgsz_test))
        print('[DATALOADER INIT]', 'Using %g dataloader workers' % dataloader.num_workers)
        # Class frequency
        if tb_writer:
            c = torch.tensor(labels_stat)  # classes
            tb_writer.add_histogram('classes', c, 0)
    return dataloader, labels_stat, testloader

def stoastic_batch_update(model, optimizer, device, loss_func, batch, ni, accumulate, ema=None, nw=None, scalar=None):
    imgs, targets = batch
    imgs = imgs.to(device)
    pred = model(imgs)
    loss = loss_func(pred, targets.to(device))  # scaled by batch_size
    if not torch.isfinite(loss):
        print('WARNING: non-finite loss, ending training ', loss.cpu().item())
        return -1
    # Backward
    loss.backward()
    # Optimize
    if ni % accumulate == 0:
        optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model)
    return loss.cpu().item()

def load_ckpt(model, optimizer, n_epochs, weights, device, results_file, rank=-1):
    # Load Model
    with torch_distributed_zero_first(rank):
        google_utils.attempt_download(weights)
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        try:
            exclude = ['anchor']  # exclude keys
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if k in model.state_dict() and not any(x in k for x in exclude)
                             and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(ckpt['model'], strict=False)
            print('Transferred %g/%g items from %s' % (len(ckpt['model']), len(model.state_dict()), weights))
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (weights, opt.cfg, weights, weights)
            raise KeyError(s) from e

        # load optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        start_epoch = ckpt['epoch'] + 1
        if n_epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (weights, ckpt['epoch'], n_epochs))
            n_epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
    return start_epoch, n_epochs, best_fitness

def save_ckpt(last, best, results_file, epoch, final_epoch, best_fitness, fi, model, optimizer):
    with open(results_file, 'r') as f:  # create checkpoint
        ckpt = {'epoch': epoch,
                'best_fitness': best_fitness,
                'training_results': f.read(),
                'model': model.module if hasattr(model, 'module') else model,
                'optimizer': None if final_epoch else optimizer.state_dict()}
    # Save last, best and delete
    torch.save(ckpt, last)
    if (best_fitness == fi) and not final_epoch:
        torch.save(ckpt, best)
    del ckpt

def train(hyp, tb_writer, opt, device):
    print(f'Hyperparameters {hyp}')
    log_dir = tb_writer.log_dir if tb_writer else 'runs/evolution'  # run directory
    wdir = str(Path(log_dir) / 'weights') + os.sep  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    results_file = log_dir + os.sep + 'results.txt'
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.local_rank
    # TODO: Init DDP logging. Only the first process is allowed to log.
    # Since I see lots of print here, the logging configuration is skipped here. We may see repeated outputs.

    # Save run settings
    with open(Path(log_dir) / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(Path(log_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Create model
    model = Resnet(opt.cfg, nc=nc).to(device)

    # create optimizer
    optimizer, accumulate, nbs = get_optimizer(model, total_batch_size, hyp)

    # load ckpt
    start_epoch, epochs, best_fitness = load_ckpt(model, optimizer, epochs, weights, device, results_file, rank)

    # create scheduler
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # DP mode
    if device.type != 'cpu' and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and device.type != 'cpu' and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        print('Using SyncBatchNorm()')

    # Exponential moving average
    ema = torch_utils.ModelEMA(model) if rank in [-1, 0] and opt.ema else None

    # DDP mode
    if device.type != 'cpu' and rank != -1:
        model = DDP(model, device_ids=[rank], output_device=rank)

    # dataloader and data_statistic
    dataloader, labels_stat, testloader = get_dataloader(hyp, train_path, test_path, batch_size, total_batch_size, opt.img_size, rank, tb_writer)
    nb = len(dataloader)  # number of batches

    # warmup scheduler
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    warmup_scheduler = get_warmup_tuner(lf, nw, nbs, total_batch_size, hyp) if opt.warmup else None

    # Model parameters
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    model.class_weights = labels_to_class_weights(labels_stat[None], nc).to(device)  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    scheduler.last_epoch = start_epoch - 1  # do not move
    if rank in [0, -1]:
        print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)

        pbar = enumerate(dataloader)
        if rank in [-1, 0]:
            print(('\n' + '%10s' * 4) % ('Epoch', 'gpu_mem', 'lr', 'cls_loss'))
            pbar = tqdm(pbar, total=nb)  # progress bar
        mloss = torch.zeros(1, device=device)  # mean losses
        for i, (imgs, targets) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # Warmup
            if warmup_scheduler and ni < nw:
                accumulate = warmup_scheduler(optimizer, epoch, ni)
            loss_scala = stoastic_batch_update(model, optimizer, device, loss_func, (imgs, targets), ni, accumulate, ema)
            if loss_scala < 0:
                return
            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_scala) / (i + 1)  # update mean losses
            if rank in [-1, 0]:
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.2g' + '%10.4g' * 1) % ('%g/%g' % (epoch, epochs - 1), mem, optimizer.param_groups[0]['lr'], *mloss)
                pbar.set_description(s)
                # Plot
                if ni < 3:
                    f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                    result = plot_train_images(images=imgs, targets=targets, fname=f)
                    if tb_writer and result is not None:
                        tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                        # tb_writer.add_graph(model, imgs)  # add model to tensorboard
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        # Only the first process in DDP mode is allowed to log or save checkpoints.
        if rank in [-1, 0]:
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:  # Calculate accuracy
                acc, loss = test.test(opt.data, 
                                      model=(ema.ema.module if hasattr(ema.ema, 'module') else ema.ema) if ema else model, 
                                      dataloader=testloader, half_test=half_test)
                # Write
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 2 % (acc, loss) + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
                # Tensorboard
                if tb_writer:
                    tags = ['train/cls_loss','metrics/accuracy', 'val/cls_loss']
                    for x, tag in zip(list(mloss) + [acc,] + list(loss), tags):
                        tb_writer.add_scalar(tag, x, epoch)
                # Update best acc
                fi = acc
                if fi > best_fitness:
                    best_fitness = fi
            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                save_ckpt(last, best, results_file, epoch, final_epoch, best_fitness, fi, model, optimizer)
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    if rank in [-1, 0]:
        # Strip optimizers
        n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
        # Finish
        plot_results(save_dir=log_dir)  # save as results.png
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    dist.destroy_process_group() if rank not in [-1, 0] else None
    torch.cuda.empty_cache()
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='models/yolov4l-mish.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyp.yaml path (optional)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/to/last.pt, or most recent run if blank.')
    parser.add_argument('--warmup', action='store_true', help='only save final checkpoint')
    parser.add_argument('--ema', action='store_true', help='only save final checkpoint')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    opt = parser.parse_args()

    last = get_latest_run() if opt.resume == 'get_last' else opt.resume  # resume from most recent run
    if last and not opt.weights:
        print(f'Resuming training from {last}')
    opt.weights = last if opt.resume and not opt.weights else opt.weights
    # if opt.local_rank in [-1, 0]:
    #     check_git_status()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    if opt.hyp:  # update hyps
        opt.hyp = check_file(opt.hyp)  # check file
        with open(opt.hyp) as f:
            hyp.update(yaml.load(f, Loader=yaml.FullLoader))  # update hyps
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = torch_utils.select_device(opt.device, apex=False, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    if device.type == 'cpu':
        mixed_precision = False
    elif opt.local_rank != -1:
        # DDP mode
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        opt.world_size = dist.get_world_size()
        assert opt.batch_size % opt.world_size == 0, "Batch size is not a multiple of the number of devices given!"
        opt.batch_size = opt.total_batch_size // opt.world_size
    print(opt)

    # Train
    if opt.local_rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(log_dir=increment_dir('runs/exp', opt.name))
    else:
        tb_writer = None
    train(hyp, tb_writer, opt, device)