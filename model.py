import torch
import torch.nn as nn

from loss_classes import ContentLoss, StyleLoss, TemporalLoss


class Normalization(nn.Module):
    def __init__(self, mean=torch.tensor([0.485, 0.456, 0.406]).to('cuda'), std=torch.tensor([0.229, 0.224, 0.225]).to('cuda')):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, style_img1, style_img2, content_img, style_weights, previous_frame, temporal_weight,
                               content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    normalization = Normalization()
    content_losses = []
    style_losses = []
    temporal_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature1 = model(style_img1).detach()
            style_loss1 = StyleLoss(target_feature1)
            model.add_module("style_loss_{}_1".format(i), style_loss1)
            style_losses.append((style_loss1, style_weights[0]))

            target_feature2 = model(style_img2).detach()
            style_loss2 = StyleLoss(target_feature2)
            model.add_module("style_loss_{}_2".format(i), style_loss2)
            style_losses.append((style_loss2, style_weights[1]))

            if previous_frame is not None and name in ['conv_1', 'conv_2']:#temporal loss on early conv layers
                warped_target = model(previous_frame).detach()
                temporal_loss = TemporalLoss(warped_target, temporal_weight)
                model.add_module("temporal_loss_{}".format(i), temporal_loss)
                temporal_losses.append(temporal_loss)


    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses, temporal_losses