import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(10, 2)

    def forward(self, x):

        x = self.fc(x)

        return x


# 创建模型
model = MyModel()

print("模型：")
print(model)


# 创建输入数据
x = torch.randn(4, 10)

print("\n输入 shape:")
print(x.shape)


# 模型计算
output = model(x)

print("\n输出 shape:")
print(output.shape)


# 训练模式
model.train()

print("\n训练模式:")
print(model.training)


# 验证模式
model.eval()

print("\n验证模式:")
print(model.training)