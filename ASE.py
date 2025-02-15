import torch
import torch.nn as nn
import torch.nn.functional as F

# ASE（Attention Squeeze-Excitation）模块
class ASE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ASE, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

# Local-Global 结合的深度估计网络
class LocalGlobalDepthNet(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(LocalGlobalDepthNet, self).__init__()
        # 局部特征提取
        self.local_conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.local_conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        # 全局特征提取
        self.global_conv1 = nn.Conv2d(in_channels, features, kernel_size=5, padding=2)
        self.global_conv2 = nn.Conv2d(features, features, kernel_size=5, padding=2)

        # ASE 注意力融合
        self.ase = ASE(features * 2)

        # 深度估计头
        self.depth_head = nn.Conv2d(features * 2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        local_features = F.relu(self.local_conv1(x))
        local_features = F.relu(self.local_conv2(local_features))

        global_features = F.relu(self.global_conv1(x))
        global_features = F.relu(self.global_conv2(global_features))

        combined_features = torch.cat([local_features, global_features], dim=1)
        attended_features = self.ase(combined_features)

        depth_map = self.depth_head(attended_features)
        depth_map = torch.sigmoid(depth_map)  # 归一化到 0-1
        return depth_map
