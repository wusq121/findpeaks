import numpy as np


def precision_recall(test, original):
    """
    计算精度
    参数：
    test: 实验后所提取出的相关指标,
    original: 实验前所提取出的相关指标
    返回：
    精度,召回率
    """
    n = test.size
    m = original.size
    tp = np.sum([ti in original for ti in test])
    return [tp/n, tp/m]
