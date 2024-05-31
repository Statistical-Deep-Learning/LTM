""" Cross Entropy w/ smoothing or soft targets

Hacked together by / Copyright 2021 Ross Wightman
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def one_hot_encode(labels, num_classes):
    # Create a tensor filled with zeros and of appropriate size
    one_hot = torch.zeros(labels.size(0), num_classes).to(labels.device)
    # Use scatter to place ones where the class label indicates
    one_hot.scatter_(1, labels.unsqueeze(1), 1.)
    return one_hot

class LabelSmoothingCrossEntropy_IB(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy_IB, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target, clust_score_input, clust_score_output, Q):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        IB_Loss = self.get_IB_Loss(target, clust_score_input, clust_score_output, Q)
        return loss.mean(), IB_Loss
    
    def get_IB_Loss(self, target, clust_score_input, clust_score_output, Q):
        # clust_score_input: batch_size x cluster_num
        # clust_score_output: batch_size x cluster_num
        # Q: cluster_num x cluster_num
        class_num = Q.shape[1]
        one_hot_target = one_hot_encode(target, class_num)

        term_1 = torch.sum(clust_score_output.unsqueeze(2) * clust_score_input.unsqueeze(1) * torch.log(clust_score_input.unsqueeze(1)),dim=(-2,-1))
        term_2 = torch.sum(clust_score_output.unsqueeze(2) * one_hot_target.unsqueeze(1) * torch.log(Q.unsqueeze(0)),dim=(-2,-1))
        bound = term_1 - term_2
        return bound.mean()
    
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
