import torch
import torch.nn as nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


net = SimpleNet()
input_tensor = torch.randn(1, 1, 28, 28)  # 例如，一个单通道的28x28图像
writer = SummaryWriter('logs_train2')  # 指定TensorBoard日志目录

with open('network_structure.txt', 'w') as f:
    f.write(str(net))  # 直接将网络结构写入文件

writer.add_graph(net, (input_tensor))
writer.close()




