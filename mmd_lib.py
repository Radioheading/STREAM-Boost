import torch

def guassian_kernel(src,tgt,kernel_mul = 2,kernel_num = 5,fix_sigma = None):
    n_samples = int(src.size()[0])+int(tgt.size()[0])
    total = torch.cat([src,tgt],dim = 0)# 按列合并 (n_samples,feature_dim)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd_rbf_(src,tgt,sigma_list):
    batch_size = src.shape[0]
    kernels = guassian_kernel(src, tgt,
                              kernel_mul=2, kernel_num=5, fix_sigma=None)
    X_size = src.shape[0]
    XX = kernels[:X_size, :X_size].mean()
    YY = kernels[X_size:, X_size:].mean()
    XY = kernels[:X_size, X_size:].mean()
    return XX + YY - 2 * XY