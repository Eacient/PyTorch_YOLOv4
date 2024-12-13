import argparse
from copy import deepcopy

from resnet.models.base import *
# from base import *

def init_weights(m:nn.Module, gain=0.5, eps=1e-4, inplace=True):
    t = type(m)
    # print(t)
    # 1. conv weights
    if t is nn.Conv2d:
        # print('conv weight')
        # nn.init.xavier_normal_(m.weight, gain=gain)
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 2. linear weights
    elif t is nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=gain)
    # 3. batchnorm eps
    elif t is nn.BatchNorm2d:
        m.eps = eps
        # m.momentum = 0.03
    # 4. relu inplace
    elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        m.inplace = inplace

def get_opt_param_groups_cnn(model:nn.Module, wd:float):
    scale_weights = [] # now only consider bn named in ReLUConv
    mm_weights = []
    bias = []
    other = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        if '.weight' in k:
            if '.bn' in k:
                scale_weights.append(v)
            else:
                mm_weights.append(v)
        elif '.bias' in k:
            bias.append(v)
        else:
            other.append(v)
    print('[OPTIMIZER GROUPS]: %g .bias, %g scale.weight, %g mm.weight, %g other' % (len(bias), len(scale_weights), len(mm_weights), len(other)))
    base_group = {'params': scale_weights + other}
    wd_group = {'params': mm_weights, 'weight_decay': wd}
    bias_group = {'params': bias}
    return [base_group, wd_group, bias_group]

def get_warmup_tuner(lf, nw, nbs, total_batch_size, hyp):
    print('[WARMUP SCHEDULER INIT]', f'nw: {nw}, nbs: {nbs}, tbs: {total_batch_size}')
    def warmup_scheduler(optimizer, epoch, ni):
        xi = [0, nw]  # x interp
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])
        accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
        return accumulate
    return warmup_scheduler

class Resnet(nn.Module):
    def __init__(self, cfg='resnet50.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Resnet, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        # Define model
        if nc and nc != self.yaml['nc']:
            print('Overriding %s nc=%g with nc=%g' % (cfg, self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_resnet_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        
        conv_count = 0
        for k, v in self.model.named_modules():
            # print(k, type(v), type(v) is ReLUConv)
            if 'path' not in k and type(v) is ReLUConv:
                conv_count += 1
        print('[MODEL INIT]', 'stem_convs:', conv_count)

        # Init weights, biases
        self.apply(lambda m : init_weights(m, gain=0.5, eps=1e-4, inplace=True))
        self.info()
        print('')

    def forward(self, x):
        return self.model(x)

    def info(self):  # print model information
        torch_utils.model_info(self)

def parse_resnet_model(d, ch):  # model_dict, input_channels(3)
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    nc, gd, gw = d['nc'], d['depth_multiple'], d['width_multiple']

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 2) if n > 2 else n  # depth gain
        if m in [nn.Conv2d, Conv, ReLUConv, ResBottleneckLayer, ResLayer, ResBlock, ResBottleneck, nn.Linear]:
            c1, c2 = ch[f], eval(args[0]) if isinstance(args[0], str) else args[0]
            c2 = make_divisible(c2 * gw, 8) if i != len(d['backbone']) + len(d['head']) - 1 else c2
            args = [c1, c2, *args[1:]]
            if m in [ResBottleneckLayer, ResLayer]:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            c2 = ch[f]
            args = [ch[f]]
        elif m is nn.AdaptiveAvgPool2d:
            c2 = ch[f]
            args = [(1,1)]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='resnet18.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Resnet(opt.cfg).to(device)
    model.train()

    print(model)
