import torch
import torch.nn


class FlappyQNet(torch.nn.Module):
    '''
    用于FlappyBird游戏的QNetwork
    '''

    def __init__(self):
        '''
        创建神经网络，在创建类实例时会自动调用
        模型结构：卷积层-卷积层-全连接层-全连接层
        '''
        super(FlappyQNet, self).__init__()

        self.actions = 2  # 输出的动作数量

        """
        torch.nn.Conv2d()
        in_channels: 第一个参数，输入的通道数
        out_channels: 第二个参数，输出的通道数
        kernel_size: 卷积核的大小，一般采用长宽相同的卷积核，所以kernel_size=8代表一个8*8的卷积核
        stride: 卷积核在图像窗口上平移的步长
        padding: 在图像四周（用0）填充多大的尺寸。例：原图像32*32，padding=1，填充后的图像尺寸34*34

        torch.nn.ReLU()
        inplace: 计算的值是否覆盖输入的值。若inplace=True，计算后输入的原始变量会被新值覆盖，这样做能省内存
        """
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        # 经过第2个卷积层之后，tensor后3个维度的尺寸如map_size所示
        self.map_size = (64, 16, 9)

        """
        torch.nn.Linear()
        线性层（全连接层），将输入与权重矩阵相乘并加上偏置，然后通过激活函数进行非线性变换
        in_features: 第一个参数。输入的神经元个数
        out_features: 第二个参数。输出的神经元个数
        """
        self.fc1 = torch.nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 256)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.fc2 = torch.nn.Linear(256, self.actions)

    def forward(self, observation):
        '''
        根据当前的观测o，给出对应的Q值
        '''

        """
        before go into the net, size: torch.Size([1, 4, 128, 72])
        after self.conv1, size: torch.Size([1, 32, 32, 18])
        after self.relu1, size: torch.Size([1, 32, 32, 18])
        after self.conv2, size: torch.Size([1, 64, 16, 9])
        after self.relu2, size: torch.Size([1, 64, 16, 9])
        after out.view(out.size()[0], -1), size: torch.Size([1, 9216])
        after self.fc1, size: torch.Size([1, 256])
        after self.relu3, size: torch.Size([1, 256])
        after self.fc2, size: torch.Size([1, 2])
        """
        out = self.conv1(observation)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        """
        用torch.view()改变tensor尺寸，不改变数据
        改变前后各个维度上尺寸的乘积相等（数据量相等）
        -1 代表这个维度的尺寸由电脑自动计算
        注意改变尺寸前后的tensor共享同一块内存
        """
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        q_value = self.fc2(out)
        return q_value


class FlappyDuelingQNet(torch.nn.Module):
    '''
    用于FlappyBird游戏的QNetwork，采用Dueling DQN理论设计网络结构
    '''

    def __init__(self):
        '''
        创建神经网络，在创建类实例时会自动调用
        模型结构：卷积层-卷积层-全连接层-两个分开的全连接层
        '''
        super(FlappyDuelingQNet, self).__init__()

        self.actions = 2  # 输出的动作数量

        """
        torch.nn.Conv2d()
        in_channels: 第一个参数，输入的通道数
        out_channels: 第二个参数，输出的通道数
        kernel_size: 卷积核的大小，一般采用长宽相同的卷积核，所以kernel_size=8代表一个8*8的卷积核
        stride: 卷积核在图像窗口上平移的步长
        padding: 在图像四周（用0）填充多大的尺寸。例：原图像32*32，padding=1，填充后的图像尺寸34*34

        torch.nn.ReLU()
        inplace: 计算的值是否覆盖输入的值。若inplace=True，计算后输入的原始变量会被新值覆盖，这样做能省内存
        """
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU(inplace=True)
        # 经过第2个卷积层之后，tensor后3个维度的尺寸如map_size所示
        self.map_size = (64, 16, 9)

        """
        torch.nn.Linear()
        线性层（全连接层），将输入与权重矩阵相乘并加上偏置，然后通过激活函数进行非线性变换
        in_features: 第一个参数。输入的神经元个数
        out_features: 第二个参数。输出的神经元个数
        """
        self.fc1 = torch.nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 256)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.fc2_a = torch.nn.Linear(256, self.actions)
        self.fc2_v = torch.nn.Linear(256, 1)

    def forward(self, observation):
        '''
        根据当前的观测o，给出对应的Q值
        '''

        """
        before go into the net, size: torch.Size([1, 4, 128, 72])
        after self.conv1, size: torch.Size([1, 32, 32, 18])
        after self.relu1, size: torch.Size([1, 32, 32, 18])
        after self.conv2, size: torch.Size([1, 64, 16, 9])
        after self.relu2, size: torch.Size([1, 64, 16, 9])
        after out.view(out.size()[0], -1), size: torch.Size([1, 9216])
        after self.fc1, size: torch.Size([1, 256])
        after self.relu3, size: torch.Size([1, 256])
        after self.fc2_v, size: torch.Size([1, 1])
        after self.fc2_a, size: torch.Size([1, 2])
        q_value size: torch.Size([1, 2])
        """
        out = self.conv1(observation)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        """
        用torch.view()改变tensor尺寸，不改变数据
        改变前后各个维度上尺寸的乘积相等（数据量相等）
        -1 代表这个维度的尺寸由电脑自动计算
        注意改变尺寸前后的tensor共享同一块内存
        """
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out_v = self.fc2_v(out)
        out_a = self.fc2_a(out)
        q_value = out_v + out_a - torch.mean(out_a, dim=1, keepdim=True)
        return q_value
