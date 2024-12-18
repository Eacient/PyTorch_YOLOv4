import torch
from torch import nn
import torch_pruning as tp

import sys
sys.path.append('.')
from resnet.models.resnet import Resnet

model = Resnet('resnet/models/resnet18t-ds.yaml', nc=2).eval()
# print(model)

# 1. Build dependency graph for a resnet18. This requires a dummy input for forwarding
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

for group in DG.get_all_groups(ignored_layers=[model.model[0].conv, model.model[8]], root_module_types=[nn.Conv2d, nn.Linear]):
    idxs = [2,4,6] # your pruning indices
    group.prune(idxs=idxs)
    print(group)

# # 2. To prune the output channels of model.conv1, we need to find the corresponding group with a pruning function and pruning indices.
# group = DG.get_pruning_group(model.model[0].conv, tp.prune_conv_out_channels, idxs=[2, 6, 9])
# print(group.details())

# # 3. Do the pruning
# if DG.check_pruning_group(group): # avoid over-pruning, i.e., channels=0.
#     group.prune()
    
# # 4. Save & Load
# model.zero_grad() # clear gradients to avoid a large file size
# torch.save(model, 'model.pth') # !! no .state_dict here since the structure has been changed after pruning
# model = torch.load('model.pth') # load the pruned model