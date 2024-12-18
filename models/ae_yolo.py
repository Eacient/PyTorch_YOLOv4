import argparse
from copy import deepcopy
# import sys
# sys.path.append('.')
from models.yolo import *
from models.experimental import *
from utils.utils import compute_loss

def parse_model_ae(d, ch):  # model_dict, input_channels(3)

    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    save, c2 = [], ch[-1]  # layers, savelist, ch out

    backbone_layers, c2 = parse_head(d, ch, c2, save, 'backbone', gd, gw, no, anchors, nc)
    # print(ch, save, c2)
    head_layers, c2 = parse_head(d, ch, c2, save, 'head', gd, gw, no, anchors, nc)
    # print(ch, save, c2)

    layers = backbone_layers + head_layers

    rec_layers, c2 = parse_head(d, ch, c2, save, 'rec-head', gd, gw, no, anchors, nc)
    feat_layers, c2 = parse_head(d, ch, c2, save, 'feat-head', gd, gw, no, anchors, nc)

    return nn.Sequential(*layers), nn.Sequential(*rec_layers), nn.Sequential(*feat_layers), sorted(save)

class AEModel(Model):
    def __init__(self, cfg='models/yolov4s-ae.yaml', ch=3, nc=None):
        super(AEModel, self).__init__(cfg, ch, nc)
        _, self.rec_head, self.feat_head, self.save = parse_model_ae(deepcopy(self.yaml), ch=[ch])
        torch_utils.initialize_weights(self)
        self.info()
        print('')
    
    def forward(self, x, augment=False, profile=False):
        if augment:
            assert not self.training
        if self.training:
            x, y = self.forward_once(x, profile, keep_mid=True)
            rec_loss = self.forward_head(x, y, self.rec_head)
            feat_loss = self.forward_head(x, y, self.feat_head)
            return x, rec_loss, feat_loss
        else:
            if augment:
                img_size = x.shape[-2:]  # height, width
                s = [0.83, 0.67]  # scales
                y = []
                for i, xi in enumerate((x,
                                        torch_utils.scale_img(x.flip(3), s[0]),  # flip-lr and scale
                                        torch_utils.scale_img(x, s[1]),  # scale
                                        )):
                    # cv2.imwrite('img%g.jpg' % i, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])
                    y.append(self.forward_once(xi)[0])

                y[1][..., :4] /= s[0]  # scale
                y[1][..., 0] = img_size[1] - y[1][..., 0]  # flip lr
                y[2][..., :4] /= s[1]  # scale
                return torch.cat(y, 1), None  # augmented inference, train
            else:
                return self.forward_once(x, profile)  # single-scale inference, train
    
    def forward_head(self, x, y, mod):
        for m in mod:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

def compute_loss_ae(pred, rec_loss, feat_loss, masks, targets, model):
    loss, loss_item = compute_loss(pred, targets, model)
    h = model.hyp

    bs = rec_loss.shape[0] // 2
    feat_loss = (feat_loss.reshape(bs, -1) * masks[:, None]).mean().reshape(1)
    rec_loss = (rec_loss * masks.repeat([2])[:, None, None, None]).mean().reshape(1)
    rec_loss *= h['rec']
    feat_loss *= h['feat']

    loss = loss + rec_loss*bs + feat_loss*bs

    return loss, torch.cat((loss_item, rec_loss.detach(), feat_loss.detach()))

if __name__ == "__main__":
    model = AEModel()
    model(torch.zeros(2, 3, 256, 256))