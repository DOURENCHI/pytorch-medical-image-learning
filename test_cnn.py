import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

    def forward(self, x):

        x = self.conv1(x)
        print("Conv:", x.shape)

        x = self.relu(x)
        print("ReLU:", x.shape)

        x = self.pool(x)
        print("Pool:", x.shape)

        return x


model = SimpleCNN()

x = torch.randn(1, 1, 512, 512)

output = model(x)

print("最终输出:", output.shape)