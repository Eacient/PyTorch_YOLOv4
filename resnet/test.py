import argparse
from sklearn import metrics

from models.experimental import *
from utils.datasets import *


def test(data,
         model=None,
         dataloader=None,
         half_test=False):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    # Half
    half = half_test and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    nc = int(data['nc'])  # number of classes

    stat = []
    s = ('%20s' + '%12s' * (2+nc*2)) % ('Class', 'acc', 'val_loss', *(f'P_{i}' for i in range(nc)), *(f'R_{i}' for i in range(nc)))
    def loss_func(pred, target):
        return nn.functional.cross_entropy(pred, target)
    loss = torch.zeros(1, device=device)
    for img, targets in tqdm(dataloader, desc=s):
        img = img.to(device, non_blocking=True)
        targets = targets.to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            pred = model(img)  # inference and training outputs

            # Compute loss
            if training:  # if model has loss hyperparameters
                loss_c = loss_func(pred, targets)
                loss += loss_c
                # print(loss_c)
        
        stat.append(np.stack([pred.argmax(1).cpu().numpy(), targets.cpu().numpy()], axis=1))

    stats = np.concatenate(stat, 0)
    acc = metrics.accuracy_score(stats[:,1], stats[:,0])
    cls_metrics = []
    cls_metrics += metrics.precision_score(stats[:,1], stats[:,0], average=None).tolist()
    cls_metrics += metrics.recall_score(stats[:,1], stats[:,0], average=None).tolist()
    loss = loss / len(dataloader)
    # Print results
    pf = '%20s' + '%12.3g' * (2+nc*2)  # print format
    print(pf % ('all', acc, loss, *cls_metrics))

    # Return results
    model.float()  # for training
    return acc, loss