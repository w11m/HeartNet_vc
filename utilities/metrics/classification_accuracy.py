#============================================
__author__ = "Sachin Mehta"
__maintainer__ = "Sachin Mehta"
#============================================

import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).long().expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).long().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
