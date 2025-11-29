# 看代码之前需对照resnet详解,有助于理解代码
import torch
import torch.nn as nn

#resnet18-layer and resnet34-layer对应的残差块
class BasicBlock(nn.Module):
    # 在18-layer和34-layer的卷积中特征图的channel（通道数）没变，所以对应的expansion为1
    # 目的是用来调整卷积核的深度（channel），从而使得特征图的深度与卷积核的深度一致，因为残差连接要求输入和输出的形状（H/W/ 通道数）一致，downsample 用于对齐形状，
    # 故他们所对应的H、W和channel要相同
    expansion=1
    # in_channel:输入深度；out_channel:输出深度；stride:步长，默认值为1；downsample:下采样
    def __init__(self, in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        # 3*3(kernel_size)卷积，步长(stride)为输入步长（若为1，则图像大小不变，若为2，则图像大小缩减一半），该行代码目的为提取特征,bia：偏置
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                             kernel_size=3,stride=stride,padding=1,bias=False)
        # 利用BN来加快模型训练速度、缓解梯度消失、防止过拟合，基本上只要用了卷积操作就要用一次BN
        self.bn1=nn.BatchNorm2d(out_channel)
        # Relu是用来将矩阵中数字<0的数据改成0，>0的数据保持不变，通过给模型引入非线性（Relu），
        # 让模型能拟合复杂任务
        self.relu=nn.ReLU()

        # 以下代码都是根据Resnet详解中的18-layer与34-layer中存在的卷积数据进行重复使用
        # （例如[(3*3,64)
        #       (3*3,64)]
        # 实际上就是进行2次卷积，3*3的卷积，64的channel
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                             kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.downsample=downsample
    # 通过前馈网络调用初始化好的卷积
    def forward(self,x):
        # 保留原图像的矩阵
        identity=x
        # 看是否需要调整图像的大小或深度,如果不需要调整,则下采样为None
        if self.downsample is not None:
            identity=self.downsample(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

    # 残差连接的核心操作
    # 目的是为了实现残差学习，降低训练难度，缓解梯度消失，保留原始数据
    # 如果通过卷积前的图像与卷积后的图像的H,W,channel与卷积处理后的图像一致,那么直接将处理前图像的矩阵+卷积处理后的图像矩阵就可以了
        out+=identity
    #注意:一定要最后进行relu
    #因为要保护残差连接的恒等梯度传递,让残差更灵活
        out=self.relu(out)
        return out

# resnet 50-layer,resnet 101-layer和resnet 152-layer对应的残差块
class Bottleneck(nn.Module):
    # 由resnet详解图片可以知道conv2_x的输入channel是64,但conv2_x最后一层的channel是256,是输入图像的channel的4倍
    # 而expansion就是用来调整这个参数的
    expansion=4
    # 与BasicBlock一致
    # in_channel:输入深度；out_channel:输出深度；stride:步长，默认值为1；downsample:下采样
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(Bottleneck,self).__init__()
        # 1*1(kernel_size)卷积核，步长(stride)为1，bia:偏置，目的是提取图像特征
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                             kernel_size=1,bias=False,stride=1)
        # BN用于防止进行卷积操作后的梯度消失和过拟合，加快训练速度
        self.bn1=nn.BatchNorm2d(out_channel)
        # 非线性处理，让模型能够处理复杂的任务
        self.relu=nn.ReLU()
        # 3*3卷积核
        self.conv2=nn.Conv2d(in_channels=out_channel,out_channels=out_channel,
                             kernel_size=3,bias=False,stride=stride)
        # 非线性处理
        self.bn2=nn.BatchNorm2d(out_channel)
        self.conv3=nn.Conv2d(in_channels=out_channel,out_channels=out_channel*self.expansion,
                             kernel_size=1,bias=False,stride=1)
        self.bn3=nn.BatchNorm2d(out_channel*self.expansion)
        # inplace=True:在原处理后的图像上进行覆盖输入，节约内存
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample
    # 通过前馈网络调用初始化好的卷积
    def forward(self,x):
        identity=x
        if self.downsample is not None:
            identity=self.downsample(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out+=identity
        out=self.relu(out)

        return out

class ResNet(nn.Module):
    # block:block为BasicBlock或Bottleneck(resnet18-layer and resnet34-layer选择BasicBlock,
    #                                    resnet 50-layer,resnet 101-layer和resnet 152-layer则选择Bottleneck)
    # blcok_num:数组,对应值为resnet详解中的数据,例如34-layer,图片中是conv2_x:*3,conv3_x:*4,conv4_x:*6,conv5_x:*3,
    #                                                         则对应数组为[3,4,6,3]
    # num_classes:训练集的个数
    # include_top:作用是可以在该网络的基础上搭建其他网络
    def __init__(self,block,block_num,num_classes=1000,include_top=True):
        super(ResNet,self).__init__()
        self.include_top=include_top

        # in_channel对应将要用conv2_x处理的图像的channel
        self.in_channel=64
        # 定义conv1
        # 输入图像的channel=3表示彩色图像对应RGB,7*7卷积核,步长为2
        # padding是计算出来的,主要目的是为了通过卷积处理后能调整图像到一个适合的大小,
        # padding的计算公式：out=[(input+2*padding-kernel_size)/stride]+1
        self.conv1=nn.Conv2d(3,out_channels=self.in_channel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_channel)
        self.relu=nn.ReLU(inplace=True)

        #定义conv2_x,3*3卷积核,步长为2,padding为1
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # block为BasicBlock或Bottleneck中的一个,具体要看自身选择,64是resnet中conv2_x对应的第一个卷积核对应的channel
        self.layer1=self._make_layer(block,64,block_num[0])
        #定义conv3_x
        # block为BasicBlock或Bottleneck中的一个,具体要看自身选择,128是resnet中conv3_x对应的第一个卷积核对应的channel
        self.layer2=self._make_layer(block,128,block_num[1],stride=2)
        #定义conv4_x
        # block为BasicBlock或Bottleneck中的一个,具体要看自身选择,256是resnet中conv4_x对应的第一个卷积核对应的channel
        self.layer3=self._make_layer(block,256,block_num[2],stride=2)
        # 定义conv5_x
        # block为BasicBlock或Bottleneck中的一个,具体要看自身选择,512是resnet中conv5_x对应的第一个卷积核对应的channel
        self.layer4=self._make_layer(block,512,block_num[3],stride=2)

        # include_top: 作用是可以在该网络的基础上搭建其他网络,对应的值为BOOL值(True或False)
        if self.include_top:
        # 平均池化
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        # 全连接
            self.fc=nn.Linear(512*block.expansion,num_classes)
        # 卷积层初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
    # 这里的block_num是数值，不是数组
    def _make_layer(self,block,channel,block_num,stride=1):
        downsample = None
        # 如果步长!=1就说明图像大小(H,W)发生变化了,那么对应的原图像也要发生变化(因为out += identity,矩阵中只有相同大小的矩阵才能相加)
        # 如果深度与channel*残差块中的expansion不同(BasicBlock固定为1,Bottleneck中固定为4),说明深度发生变化了
        if stride!=1 or self.in_channel!=channel*block.expansion:
            # 利用Sequential容器将要对原图像处理的卷积封装
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channel,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )

        layers=[]
        # 将要使用的残差块插入layers数组中
        layers.append(block(self.in_channel,channel,stride=stride,downsample=downsample))
        # 调整输入深度的大小
        self.in_channel=channel*block.expansion
        # 循环从1开始是因为已经插入过1个残差块了
        # block_num 是当前 stage 的残差块重复次数（resnet详解图中那个*3，*4，*6，*3就是这里的值，要重复多次的运行残差块）
        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))
        # 将layers数组中的数据再封装到Sequential容器中
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        if self.include_top:
            x=self.avgpool(x)
            x=torch.flatten(x,1)
            x=self.fc(x)
        return x
# num_classes表示数据集的种类，官方给出的是1000，
# include_top表示是否要在该网络上再搭建更复杂的网络
def resnet18(num_classes=1000,include_top=True):
    return ResNet(BasicBlock,[2,2,2,2],num_classes,include_top)
def resnet34(num_classes=1000,include_top=True):
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)
def resnet50(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,6,3],num_classes,include_top=include_top)
def resnet101(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,4,23,3],num_classes,include_top=include_top)
def resnet152(num_classes=1000,include_top=True):
    return ResNet(Bottleneck,[3,8,36,3],num_classes,include_top=include_top)

# 测试resnet代码：
# if __name__ == '__main__':
#     # input_tensor=torch.randn(1,3,224,224)
#     # model=resnet34()
#     # output=model(input_tensor)
#     # print(output.shape)
#     x=torch.randn(1,3,224,224)
#     model=resnet34()
#     outout=model(x)
#     print(outout.shape)