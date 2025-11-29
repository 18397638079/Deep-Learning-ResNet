import torch
import torch.nn as nn

"""
ResNet 实现（支持18/34/50/101/152层）
核心原理：残差连接（Residual Connection），通过跨层恒等映射缓解梯度消失，降低深层网络训练难度
结构说明：
- BasicBlock：适用于ResNet18/34，由2个3×3卷积组成，通道扩张系数expansion=1
- Bottleneck：适用于ResNet50/101/152，由1×1+3×3+1×1卷积组成，通道扩张系数expansion=4
- 整体结构：Conv1 → MaxPool → Conv2_x → Conv3_x → Conv4_x → Conv5_x → AvgPool → FC
"""


# ResNet18/34 对应的基础残差块（2个3×3卷积）
class BasicBlock(nn.Module):
    # 通道扩张系数：输出通道数 = 基础通道数 × expansion
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        基础残差块初始化
        Args:
            in_channel (int): 输入特征图的通道数
            out_channel (int): 基础输出通道数（最终输出=out_channel×expansion）
            stride (int): 第一个3×3卷积的步长（控制特征图尺寸），默认1
            downsample (nn.Sequential): 恒等映射的下采样模块（用于对齐输入/输出形状），默认None
        """
        super(BasicBlock, self).__init__()
        # 第一个3×3卷积：提取空间特征，步长控制尺寸
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False  # BN层已包含偏置，卷积层无需偏置
        )
        self.bn1 = nn.BatchNorm2d(out_channel)  # 批量归一化：加速训练、缓解梯度消失、防止过拟合
        self.relu = nn.ReLU(inplace=True)  # 非线性激活：引入非线性，提升模型表达能力

        # 第二个3×3卷积：进一步提取特征，步长固定为1（尺寸不变）
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample  # 下采样模块：对齐残差连接的形状（H/W/通道数）

    def forward(self, x):
        """前向传播：残差连接核心逻辑"""
        identity = x  # 保存原始输入（恒等映射）
        # 若需要下采样，则对原始输入做形状对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差路径：卷积→BN→激活→卷积→BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：原始输入 + 残差路径输出（核心！缓解梯度消失）
        out += identity
        out = self.relu(out)  # 最后激活：保护恒等梯度传递，提升灵活性

        return out


# ResNet50/101/152 对应的瓶颈残差块（1×1+3×3+1×1卷积）
class Bottleneck(nn.Module):
    # 通道扩张系数：输出通道数 = 基础通道数 × 4
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        """
        瓶颈残差块初始化（减少参数量，提升计算效率）
        Args:
            in_channel (int): 输入特征图的通道数
            out_channel (int): 基础输出通道数（最终输出=out_channel×4）
            stride (int): 3×3卷积的步长，默认1
            downsample (nn.Sequential): 恒等映射的下采样模块，默认None
        """
        super(Bottleneck, self).__init__()
        # 1×1卷积：降维（减少后续3×3卷积的参数量）
        self.conv1 = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        # 3×3卷积：空间特征提取，步长控制尺寸
        self.conv2 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 1×1卷积：升维（恢复通道数，最终输出=out_channel×4）
        self.conv3 = nn.Conv2d(
            in_channels=out_channel,
            out_channels=out_channel * self.expansion,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.downsample = downsample

    def forward(self, x):
        """前向传播：瓶颈残差块的残差连接"""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差路径：1×1降维 → 3×3特征提取 → 1×1升维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000, include_top=True):
        """
        ResNet主网络初始化
        Args:
            block (nn.Module): 残差块类型（BasicBlock/Bottleneck）
            block_nums (list): 各stage的残差块重复次数，如ResNet34=[3,4,6,3]
            num_classes (int): 分类任务的类别数，默认1000（ImageNet）
            include_top (bool): 是否包含顶层的AvgPool+FC层（用于迁移学习），默认True
        """
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # Conv2_x的输入通道数

        # 初始卷积层：7×7大卷积核，步长2，降低特征图尺寸
        self.conv1 = nn.Conv2d(
            in_channels=3,  # 输入为RGB图像，通道数=3
            out_channels=self.in_channel,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # 最大池化：3×3核，步长2，进一步降低尺寸
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建4个stage（Conv2_x → Conv5_x）
        self.layer1 = self._make_layer(block, 64, block_nums[0])  # Conv2_x：步长1，尺寸不变
        self.layer2 = self._make_layer(block, 128, block_nums[1], stride=2)  # Conv3_x：步长2，尺寸减半
        self.layer3 = self._make_layer(block, 256, block_nums[2], stride=2)  # Conv4_x：步长2，尺寸减半
        self.layer4 = self._make_layer(block, 512, block_nums[3], stride=2)  # Conv5_x：步长2，尺寸减半

        # 顶层分类器（AvgPool + FC）
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化：输出固定为1×1
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层：映射到类别数

        # 卷积层参数初始化（Kaiming正态分布，适配ReLU激活）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        """
        构建单个stage（由多个残差块组成）
        Args:
            block (nn.Module): 残差块类型
            channel (int): 基础通道数
            block_num (int): 残差块重复次数
            stride (int): 第一个残差块的步长，默认1
        Returns:
            nn.Sequential: 组装好的stage模块
        """
        downsample = None
        # 需下采样的场景：步长≠1（尺寸变化） 或 输入通道≠输出通道（通道数变化）
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                # 1×1卷积：对齐通道数和尺寸
                nn.Conv2d(
                    self.in_channel,
                    channel * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = []
        # 第一个残差块：可能包含下采样（对齐形状）
        layers.append(block(self.in_channel, channel, stride=stride, downsample=downsample))
        # 更新输入通道数（后续残差块的输入=当前输出）
        self.in_channel = channel * block.expansion
        # 剩余残差块：步长固定为1，无需下采样
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        """ResNet前向传播"""
        # 初始卷积+池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4个stage
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 顶层分类器
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)  # 展平：(batch, C, 1, 1) → (batch, C)
            x = self.fc(x)

        return x


# 不同层数的ResNet构建函数
def resnet18(num_classes=1000, include_top=True):
    """构建ResNet18模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, include_top)


def resnet34(num_classes=1000, include_top=True):
    """构建ResNet34模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    """构建ResNet50模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, include_top)


def resnet101(num_classes=1000, include_top=True):
    """构建ResNet101模型"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, include_top)


def resnet152(num_classes=1000, include_top=True):
    """构建ResNet152模型"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, include_top)


# 测试代码：验证模型输出形状
if __name__ == '__main__':
    # 输入：batch_size=1，3通道，224×224（ImageNet标准输入）
    x = torch.randn(1, 3, 224, 224)
    model = resnet34()
    output = model(x)
    print(f"ResNet34输出形状: {output.shape}")  # 预期：(1, 1000)