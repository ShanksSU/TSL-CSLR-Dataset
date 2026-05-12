import gc
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, 1, 1),
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 3, T, H, W]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])   # [N*T, C, H, W]

        x = self.avgpool(x)
        x = x.flatten(1)                    # [N*T, C]
        x = self.dropout(x)
        x = self.fc(x)
        return x


def resnet18(**kwargs):
    do_load_pretrained = kwargs.pop('pretrained', True)
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if do_load_pretrained:
        print("Loading ImageNet pretrained weights for ResNet18...")
        ckpt = model_zoo.load_url(model_urls['resnet18'], map_location='cpu')
        for k in list(ckpt.keys()):
            if 'conv' in k or 'downsample.0.weight' in k:
                ckpt[k] = ckpt[k].unsqueeze(2)
        model.load_state_dict(ckpt, strict=False)
        del ckpt
        gc.collect()
    return model


def resnet34(**kwargs):
    do_load_pretrained = kwargs.pop('pretrained', True)
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if do_load_pretrained:
        print("Loading ImageNet pretrained weights for ResNet34...")
        ckpt = model_zoo.load_url(model_urls['resnet34'], map_location='cpu')
        for k in list(ckpt.keys()):
            if 'conv' in k or 'downsample.0.weight' in k:
                ckpt[k] = ckpt[k].unsqueeze(2)
        model.load_state_dict(ckpt, strict=False)
        del ckpt
        gc.collect()
    return model