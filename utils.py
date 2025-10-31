import numpy as np
from sklearn.metrics import confusion_matrix

#提供了模型训练和评估过程中需要的各种工具函数，主要包括性能指标计算，进度监控和参数处理等功能
class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()
"""
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
"""
"""
def output_metric(tar, pre, num_classes):
    # tar, pre: numpy arrays
    # 忽略 background (0)
    mask = tar != 0
    tar_v = tar[mask]
    pre_v = pre[mask]
    # 若 tar_v 最大为 num_classes (1..C)，转为 0..C-1
    if tar_v.max() == num_classes:
        tar_v = tar_v - 1

    labels = list(range(num_classes))
    matrix = confusion_matrix(tar_v, pre_v, labels=labels)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
"""
def output_metric(tar, pre, num_classes):
    """清理后的版本，保留必要信息"""
    tar = np.array(tar)
    pre = np.array(pre)

    # 统一标签映射
    if np.min(tar) == 0 and np.max(tar) == num_classes - 1:
        tar_mapped, pre_mapped, labels = tar, pre, list(range(num_classes))
    elif np.min(tar) == 1 and np.max(tar) == num_classes:
        tar_mapped, pre_mapped = tar - 1, pre - 1
        labels = list(range(num_classes))
    else:
        labels = sorted(np.unique(np.concatenate([tar, pre])))
        tar_mapped, pre_mapped = tar, pre

    matrix = confusion_matrix(tar_mapped, pre_mapped, labels=labels)
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum_val = 0  # 避免覆盖 Python 内置函数 sum
    AA = np.zeros([shape[0]], dtype=np.float32)

    for i in range(shape[0]):
        number += matrix[i, i]
        denom = np.sum(matrix[i, :])
        if denom == 0:
            AA[i] = 0.0  # 避免 NaN
        else:
            AA[i] = matrix[i, i] / denom
        sum_val += denom * np.sum(matrix[:, i])

    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum_val / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
           w = module.weight.data
           w.clamp_(1e-6, 1)
