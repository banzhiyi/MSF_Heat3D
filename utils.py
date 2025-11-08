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
    #清理后的版本，保留必要信息
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
"""
# 在 utils.py 的 output_metric 函数中检查
"""
def output_metric(tar, pre, num_classes, dataset_name='Indian', mode='test'):
    
    统一的指标计算函数
    Args:
        tar: 真实标签
        pre: 预测标签
        num_classes: 类别数（不包括背景）
        dataset_name: 数据集名称
        mode: 'train' 或 'test'
    
    tar = np.array(tar)
    pre = np.array(pre)

    # 🚨 关键修复：测试时过滤掉背景类别（0）
    if mode == 'test':
        # 创建掩码，排除背景类别（0）
        valid_mask = tar != 0
        tar_filtered = tar[valid_mask]
        pre_filtered = pre[valid_mask]

        # 如果标签从1开始，映射到0开始
        if np.min(tar_filtered) == 1 and np.max(tar_filtered) == num_classes:
            tar_mapped = tar_filtered - 1
            pre_mapped = pre_filtered
        else:
            tar_mapped = tar_filtered
            pre_mapped = pre_filtered

    else:  # 训练模式
        # 训练时通常已经处理好了
        if np.min(tar) == 0 and np.max(tar) == num_classes - 1:
            tar_mapped, pre_mapped = tar, pre
        elif np.min(tar) == 1 and np.max(tar) == num_classes:
            tar_mapped, pre_mapped = tar - 1, pre - 1
        else:
            tar_mapped, pre_mapped = tar, pre

    # 确保标签在有效范围内
    assert np.min(tar_mapped) >= 0 and np.max(tar_mapped) < num_classes, \
        f"tar_mapped 标签越界: [{np.min(tar_mapped)}, {np.max(tar_mapped)}]，期望范围: [0, {num_classes - 1}]"
    assert np.min(pre_mapped) >= 0 and np.max(pre_mapped) < num_classes, \
        f"pre_mapped 标签越界: [{np.min(pre_mapped)}, {np.max(pre_mapped)}]，期望范围: [0, {num_classes - 1}]"

    labels = list(range(num_classes))
    matrix = confusion_matrix(tar_mapped, pre_mapped, labels=labels)
    OA, AA_mean, Kappa, AA = cal_results(matrix)

    return OA, AA_mean, Kappa, AA
"""
def output_metric(tar, pre, num_classes, dataset_name='Unknown', mode='train'):
    """
    统一的指标计算函数
    Args:
        tar: 真实标签
        pre: 预测标签
        num_classes: 类别数（不包括背景）
        dataset_name: 数据集名称
        mode: 'train' 或 'test'
    """
    tar = np.array(tar)
    pre = np.array(pre)

    # 🎯 数据集特定的处理策略
    if dataset_name == 'Indian':
        # IndianPines: 训练[0-15], 测试[0-16]需要过滤背景
        if mode == 'test':
            valid_mask = tar != 0
            tar_filtered = tar[valid_mask]
            pre_filtered = pre[valid_mask]
            # IndianPines 测试标签从1开始，需要映射到0开始
            if np.min(tar_filtered) == 1 and np.max(tar_filtered) == num_classes:
                tar_mapped = tar_filtered - 1
                pre_mapped = pre_filtered
            else:
                tar_mapped = tar_filtered
                pre_mapped = pre_filtered
        else:
            # 训练模式，通常已经处理好
            if np.min(tar) == 0 and np.max(tar) == num_classes - 1:
                tar_mapped, pre_mapped = tar, pre
            else:
                tar_mapped, pre_mapped = tar - 1, pre

    elif dataset_name == 'Berlin':
        # Berlin数据集处理逻辑
        if mode == 'train':
            # 训练模式 - 使用通用处理
            tar_mapped, pre_mapped = tar, pre
        else:
            #过滤有效类别 [1,8] 并映射到 [0,7]
            valid_mask = (tar >= 1) & (tar <= num_classes)
            tar_filtered = tar[valid_mask]
            pre_filtered = pre[valid_mask]

            # 将标签从[1,8]映射到[0,7]
            tar_mapped = tar_filtered - 1
            pre_mapped = pre_filtered

    else:  # Augsburg或其他数据集
        # 通用处理逻辑
        if mode == 'test' and 0 in tar and np.max(tar) == num_classes:
            valid_mask = tar != 0
            tar_filtered = tar[valid_mask]
            pre_filtered = pre[valid_mask]
            tar_mapped = tar_filtered - 1
            pre_mapped = pre_filtered
        elif np.min(tar) == 1 and np.max(tar) == num_classes:
            tar_mapped = tar - 1
            pre_mapped = pre
        else:
            tar_mapped, pre_mapped = tar, pre

    labels = list(range(num_classes))

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
