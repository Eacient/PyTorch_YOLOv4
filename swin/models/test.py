import torch.nn.functional as F
import torch

from torch.profiler import profile, record_function, ProfilerActivity

# b = 32
# l = 2


# device = torch.device('cuda')

# activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
# sort_by_keyword = 'cuda' + "_time_total"
# for g in [4, 2, 1]:
#     for l in [0, 1, 2, 3]:
#         for b in [16, 32, 64, 128, 256][::-1]:
#             N = int(49 * (2**l)**2)
#             c1 = int(768 / 2**l) * 4
#             c2 = int(768 / 2**l)
#         # print(b, N, c1, c2)
#         # x = torch.ones(b, N, c1).to(device)
#         # weight = torch.ones(c2, c1).to(device)
#         # bias = torch.ones(c2).to(device)

#         # with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
#         #     for _ in range(10000):
#         #         # F.linear(x, weight, bias)
#         #         # x@weight.T+bias
#         #         # g_x@g_weight+g_bias
#         #         # (x.reshape(b*N, g, c1//g).transpose(0,1)@g_weight)
#         #         # g_x@g_weight
#         #         # .transpose(0,1).reshape(b, N, g, c2//g)
#         #         # g_x@g_weight
#         #         # x@weight
#         #         # F.linear(x, weight, bias)
#         #         F.linear(x, weight)
#         # print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

#             print(b, N, g, c1, c2)
            
#             g_x = torch.ones(g, b*N, c1//g).to(device)
#             g_weight = torch.ones(g, c1//g, c2//g).to(device)
#             g_bias = torch.ones(g, 1, c2//g).to(device)

#             activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
#             with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
#                 for _ in range(10000):
#                     # (x.reshape(b*N, g, c1//g).transpose(0,1)@g_weight)
#                     # z = torch.baddbmm(g_bias, g_x, g_weight)
#                     # torch.bmm(x.reshape(b*N, g, c1//g).transpose(0,1), g_weight).transpose(0,1).reshape(b, N, g, c2//g)
#                     g_x@g_weight
#                     # x@weight
#             print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))

from torch import nn

g = 4
c1 = 96*4
c2 = 96
N = 56*56
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(g, c1//g, c2//g))
    
    def forward(self, x):
        return x@self.weight

if __name__ == "__main__":
    # ckpt = torch.load('/root/codes/mae/tiny_out/checkpoint-380.pth', map_location='cpu')
    # print(ckpt['model'].keys())
    # ['patch_embed.proj.weight']
    # ['patch_embed.proj.bias']
    model = Model()
    torch.compile(model)
    