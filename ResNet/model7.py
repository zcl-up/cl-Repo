import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# 定义ResNet块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


# 定义注意力模块
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels,  stride=1):
        super(AttentionModule, self).__init__()
        self.first_residual_block = ResidualBlock(in_channels, out_channels,
                                                  stride=stride)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.second_residual_block = ResidualBlock(out_channels, out_channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              padding= 3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.first_residual_block(x)
        x_pooled = self.pool(x)
        x_pooled = self.second_residual_block(x_pooled)
        max_out = self.mlp(self.max_pool(x_pooled))
        avg_out = self.mlp(self.avg_pool(x_pooled))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


# 构建模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attention_module1 = AttentionModule(64, 128, stride=2)#(64,128,16,16)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)#(64,128,8,8)
        self.attention_module2 = AttentionModule(128, 256, stride=2)#(64,256,4,4)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attention_module3 = AttentionModule(256, 512, stride=1)

        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(out_channels, out_channels, stride))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.attention_module1(out)

        out = self.layer2(out)
        out = self.attention_module2(out)

        out = self.layer3(out)
        out = self.attention_module3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out