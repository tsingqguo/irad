import math
import random

import numpy as np
import torch
import torch.nn as nn
import os

__all__ = [
    "ResNetLiif",
    "resnet18_liif",
    "resnet34_liif",
    "resnet50_liif",
]

from torch.utils.tensorboard import SummaryWriter

from data.LinfDataset import resize_fn, to_pixel_samples
from models.liif import LIIF
from torchvision.transforms import ToPILImage


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetLiif(nn.Module):
    def __init__(
            self,
            config,
            liif_path,
            block,
            layers,
            num_classes=10,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ):
        super(ResNetLiif, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.liif = LIIF(imnet_spec='mlp')
        # liif_path = './logs/test_imgnet100_1206/checkpoint/300.pt'
        # liif_path = './logs/test_imgnet10_1206/checkpoint/5100.pt'
        print('liif_path: ', liif_path)
        if liif_path is not None:
            # self.liif = torch.load(liif_path, map_location=torch.device('cpu'))
            state_dict = torch.load(liif_path, map_location=torch.device('cpu'))['model']
            self.liif.load_state_dict(state_dict)

        for name, param in self.liif.named_parameters():
            param.requires_grad = False

        self.augment = config.augment
        self.gt_resize = config.gt_resize
        self.sample_q = config.sample_q

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            inp = x['inp']
            coord = x['coord']
            cell = x['cell']
            x = self.liif(inp, coord, cell)
            dis_shape = int(math.sqrt(x.shape[1]))
            reshape_size = [x.shape[0], dis_shape, dis_shape, -1]
            x = x.view(reshape_size).permute(0, 3, 1, 2).contiguous()

        # with torch.no_grad():
        #     img_lr = x
        #     img_hr = x
        #
        #     if self.augment:
        #         if random.random() < 0.5:
        #             img_lr = img_lr.flip(-1)
        #             img_hr = img_hr.flip(-1)
        #
        #     if self.gt_resize is not None:
        #         img_hr = resize_fn(img_hr, self.gt_resize)
        #
        #     hr_coord, hr_rgb = to_pixel_samples(img_hr)
        #     # mask_cood, mask_rgb = to_pixel_samples(item[2])
        #     # if hr_coord.shape[0] != 1024:
        #     #     print(hr_coord)
        #     if self.sample_q is not None:
        #         sample_lst = np.random.choice(
        #             len(hr_coord), self.sample_q, replace=False)
        #         hr_coord = hr_coord[sample_lst]
        #         hr_rgb = hr_rgb[sample_lst]
        #         # mask_cood = mask_cood[sample_lst]
        #         # mask_rgb = mask_rgb[sample_lst]
        #
        #     cell = torch.ones_like(hr_coord)
        #     cell[:, 0] *= 2 / img_hr.shape[-2]
        #     cell[:, 1] *= 2 / img_hr.shape[-1]
        #
        #     inp = img_lr
        #     coord = hr_coord,
        #     cell = cell
        #     x = self.liif(inp, coord, cell)
        #     dis_shape = int(math.sqrt(x.shape[1]))
        #     reshape_size = [x.shape[0], dis_shape, dis_shape, -1]
        #     x = x.view(reshape_size).permute(0, 3, 1, 2).contiguous()

            # ToPILImage()(x[0]).show()
            # ToPILImage()(inp[0]).show()
            # torch.save(x, './images/liif_output.pt')
            # torch.save(inp, './images/inp.pt')
            # writer0 = SummaryWriter(f'./images/tensorboard')
            # writer0.add_images("img_input", inp, 0)
            # writer0.add_images("img_output", x, 0)
            # writer0.close()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(config, liif_path, arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNetLiif(config, liif_path, block, layers, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18_liif(config, liif_path=None, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        config, liif_path, "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


def resnet34_liif(config, liif_path=None, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        config, liif_path, "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )


def resnet50_liif(config, liif_path=None, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        config, liif_path, "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, **kwargs
    )
