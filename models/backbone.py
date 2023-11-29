# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from torchinfo import summary

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        print(summary(self.body, (1,3,640,640)))

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class ShufflenetBackbone(nn.Module):
    def __init__(self, name: str, return_interm_layers: bool):
        super().__init__()

        # backbone = torchvision.models.shufflenet_v2_x1_0(weights='IMAGENET1K_V1')
        backbone = getattr(torchvision.models, name)(weights='IMAGENET1K_V1')
        if name == 'shufflenet_v2_x0_5':
            self.num_channels = 1024
        elif name == 'shufflenet_v2_x1_0':
            self.num_channels = 1024
        elif name == 'shufflenet_v2_x1_5':
            self.num_channels = 1024
        elif name == 'shufflenet_v2_x2_0':
            self.num_channels = 2048

        if return_interm_layers:
            return_layers = {"conv1": "0", "maxpool": "1", "stage2": "2", "stage3": "3", "stage4": "4", "conv5": "5"}
        else:
            return_layers = {"conv5": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        print(summary(self.body, (1,3,640,640)))

        # self.conv_out = nn.Conv2d(1024, self.num_channels, kernel_size=1, padding=0, stride=1)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}

        # Copy paste from Shufflenet source code with final class head removed.
        # xs = self.backbone.conv1(xs)
        # xs = self.backbone.maxpool(xs)
        # xs = self.backbone.stage2(xs)
        # xs = self.backbone.stage3(xs)
        # xs = self.backbone.stage4(xs)
        # xs = self.backbone.conv5(xs)
        # x = x.mean([2, 3])  # globalpool
        # x = self.fc(x)
        # xs = self.conv_out(xs)

        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if "resnet" in args.backbone:
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif "shufflenet_v2" in args.backbone:
        backbone = ShufflenetBackbone(args.backbone, return_interm_layers)
    else:
        raise Exception("Unknown backbone {}".format(args.backbone))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
