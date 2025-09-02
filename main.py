# 胖墩会武术 - 20241107
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        """__init__：定义网络结构"""
        super(SimpleCNN, self).__init__()  # 作用: 在 SimpleCNN 类中，使用 self 来调用父类的 nn.Module 的初始化方法。
        # SimpleCNN   ：当前子类的名字。
        # self        ：当前实例，传入 super() 以告诉它要操作的是当前对象。
        # __init__    ：表示调用的是 nn.Module 中的初始化方法。
        
        # 定义模型的各个层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)   # 卷积层1：输入通道 3，输出通道16，卷积核大小5x5
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1, padding=0)  # 卷积层2：输入通道16，输出通道32，卷积核大小3x3
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层2

        self.fc3 = nn.Linear(6 * 6 * 36, 128)  # 全连接层3：将多维特征图展平为一维向量（6 * 6 * 36 = 1296）
        self.fc4 = nn.Linear(128, 10)  # 全连接层4（输出层）

    def forward(self, x):
        """forward：前向传播过程"""
        x = self.pool1(F.relu(self.conv1(x)))        # 卷积层1 -> 激活函数（relu） -> 池化层1
        x = self.pool2(F.relu(self.conv2(x)))        # 卷积层2 -> 激活函数（relu） -> 池化层2
        x = x.view(-1, 6 * 6 * 36)                  # 展平操作（参数-1表示自动调整size）
        x = F.relu(self.fc3(x))                # 全连接层3 -> 激活函数（relu）
        x = self.fc4(x)                        # 输出层（最后一层通常不使用激活函数）
        return x


if __name__ == "__main__":
    model = SimpleCNN()  # 模型实例化
    #############################################################################
    input_data = torch.randn(1, 3, 32, 32)  # batch_size=1, 通道数=3, 图像尺寸=32x32
    output = model(input_data)  # 前向传播
    output_forward = model.forward(input_data)  # （显示）前向传播
    #############################################################################
    print("模型结构:\n", model)
    print("模型输出（十分类）:\n", output)
    print(output_forward)

"""
模型结构:
SimpleCNN(
  (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(16, 36, kernel_size=(3, 3), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc3): Linear(in_features=1296, out_features=128, bias=True)
  (fc4): Linear(in_features=128, out_features=10, bias=True)
)
#####################################################################################
模型输出（十分类）:
tensor([[ 0.1202,  0.0287, -0.0160,  0.0384,  0.0442,  0.0127, -0.0626, -0.1158,
         -0.0499,  0.1266]], grad_fn=<AddmmBackward0>)
tensor([[ 0.1202,  0.0287, -0.0160,  0.0384,  0.0442,  0.0127, -0.0626, -0.1158,
         -0.0499,  0.1266]], grad_fn=<AddmmBackward0>)
"""
