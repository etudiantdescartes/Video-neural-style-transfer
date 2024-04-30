import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLoss(nn.Module):
    """
    Class to compute temporal loss enforcing consistency within consecutive frames
    """
    def __init__(self, target_feature, weight):
        super(TemporalLoss, self).__init__()
        self.target = target_feature.detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) * self.weight
        return input


class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input