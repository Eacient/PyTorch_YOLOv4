from copy import deepcopy

from swin.models.base import *
# from base import *
from timm.layers import trunc_normal_

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv1d): #used in shuffle_mlp
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        # m.eps = 1e-3

def init_postnorm(m):
    if isinstance(m, SwinTransformerBlock):
        nn.init.constant_(m.norm1.bias, 0)
        nn.init.constant_(m.norm1.weight, 0)
        nn.init.constant_(m.norm2.bias, 0)
        nn.init.constant_(m.norm2.weight, 0)

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_opt_param_groups(model, skip_keywords=("cpb_mlp", "logit_scale", 'relative_position_bias_table')):
    scale_weights = []
    mm_weights = []
    bias = []
    skip = []
    other = []
    for name, param in model.named_parameters():
        if not param.requires_grad: # frozen weights
            continue
        elif 'bias' in name:
            bias.append(param)
        elif 'weight' in name  or check_keywords_in_name(name, skip_keywords):
            if check_keywords_in_name(name, skip_keywords):
                skip.append(param)
                # print(name)
            elif len(param.shape) == 1 or 'norm' in name:
                scale_weights.append(param)
            else:
                mm_weights.append(param)
        else:
            other.append(param)
    print('[OPTIMIZER GROUPS]: %g bias, %g skip.weight, %g scale.weight, %g mm.weight, %g other' % (len(bias), len(skip), len(scale_weights), len(mm_weights), len(other)))
    has_decay = mm_weights + other
    no_decay = scale_weights + skip
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': bias, 'weight_decay': 0.}]

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



class Swin(nn.Module):
    def __init__(self, cfg='swin-t.yaml', ch=3, image_size=None, nc=None):
        super().__init__()
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
        if image_size and image_size != self.yaml['image_size']:
            print('Overriding %s input=%g with input=%g' % (cfg, self.yaml['image_size'], image_size))
            self.yaml['image_size'] = image_size  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        
        self.apply(init_weights)
        self.apply(init_postnorm)
        # self.load_patch_embed_pretrained()
        
        self.info()
        print('')

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x

    def info(self):  # print model information
        torch_utils.model_info(self)
        
    def load_patch_embed_pretrained(self, pretrained='/root/codes/mae/tiny_out/checkpoint-380.pth'):
        print('[MODEL INIT]', 'LOADING PATCHMEBED FROM', pretrained)
        model_dict = torch.load(pretrained, map_location='cpu')['model']
        patch_embed_dict = {k[len('patch_embed.'):]: v for k, v in model_dict.items() if 'patch_embed' in k}
        # for k in patch_embed_dict:
        #     print(k)
        self.model[0].load_state_dict(patch_embed_dict)
        self.model[0].requires_grad_(False)

def get_stochastic_dpr(d, depth_multiple, drop_path_rate):
    depth_layers = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        n = max(round(n * depth_multiple), 2) if n > 2 else n  # depth gain
        if m is SwinLayer:
            depth_layers.append(n)
    if drop_path_rate > 0:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth_layers))]  # stochastic depth decay rule
        drop_paths = [dpr[sum(depth_layers[:i_layer]):sum(depth_layers[:i_layer + 1])] for i_layer in range(len(depth_layers))]
    else:
        drop_paths = [0. for _ in range(len(depth_layers))]
    return drop_paths

def parse_model(d, ch):
    print('\n%3s%18s%3s%10s%12s  %-40s%-30s' % ('', 'from', 'n', 'params', 'flops', 'module', 'arguments'))

    # linear nc spec
    nc = d['nc']
    
    # window spec
    image_size, patch_size = to_2tuple(d['image_size']), to_2tuple(d['patch_size'])
    window_size = d['window_size']
    embed_dim = d['embed_dim']

    # drop spec
    drop, attn_drop, drop_path = d['drop'], d['attn_drop'], d['drop_path']

    # efficiency spec
    group, depth_multiple, width_multiple = d['group'], d['depth_multiple'], d['width_multiple']
    
    # init patch_resolutions and drop_paths
    drop_paths = iter(get_stochastic_dpr(d, depth_multiple, drop_path)) # iterator
    patch_resolutions = [image_size, ]
    
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        nc_corr = False
        for j, a in enumerate(args):
            if j==0 and a == 'nc':
                nc_corr = True # c2 = nc
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * depth_multiple), 2) if n > 2 else n  # depth gain
        if m in [nn.Conv2d, PatchEmbed, SwinLayer, nn.Linear]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * width_multiple, 8) if not nc_corr else c2 # not change output nc
            args = [c1, c2, *args[1:]]
            if m is PatchEmbed:
                args.insert(2, patch_resolutions[-1]) # c1, c2, image_size
                args.insert(3, patch_size) # c1, c2, image_size, patch_size
                patch_resolutions.append((patch_resolutions[-1][0] // patch_size[0], patch_resolutions[-1][1] // patch_size[1]))
            elif m is SwinLayer:
                num_heads = args[2]
                down_sample = args[3]
                if len(args) >= 5:
                    pretrained_window_size = args[4]
                    args.pop(4)
                else:
                    pretrained_window_size = 0
                # args[2] = make_divisible(num_heads * width_multiple, 2) if num_heads != 3 else 3 # verify num_heads
                args.insert(2, n) # c1, c2, n, num_heads, down_sample
                n = 1
                args.insert(5, group)
                args.insert(6, (patch_resolutions[-1], window_size, pretrained_window_size)) # c1, c2, n, window_spec num_heads, down_sample
                args.insert(7, (drop, attn_drop, next(drop_paths))) # c1, c2, n, window_spec, drop_spec, num_heads, down_sample
                # print(args)
                if down_sample:
                    patch_resolutions.append((patch_resolutions[-1][0] // 2, patch_resolutions[-1][1] // 2))
                    c2 *= 2
        elif m is nn.LayerNorm:
            c2 = ch[f]
            args = [ch[f]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        fp = m_.flops() if m in [PatchEmbed, SwinLayer] else -1
        m_.i, m_.f, m_.type, m_.np, m_.fp = i, f, t, np, fp  # attach index, 'from' index, type, number params
        print('%3s%18s%3s%10.0f% 12g  %-40s%-30s' % (i, f, n, np, fp, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        ch.append(c2)
    return nn.ModuleList(layers), sorted(save)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='swin-t-ss.yaml', help='model.yaml')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    device = torch_utils.select_device(opt.device)

    # Create model
    model = Swin(opt.cfg).to(device)
    model.train()
    
    model(torch.ones(1, 3, 224, 224).to(device))
    
    get_opt_param_groups(model)
    # def test():
    #     a = [1, 2, 3, 4]
    #     return iter(a)
    
    # it = test()
    # print(next(it))