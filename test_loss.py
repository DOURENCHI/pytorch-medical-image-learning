import torch
import torch.nn as nn


# =========================
# 1. 创建一个简单模型
# =========================

model = nn.Linear(2, 1)


# =========================
# 2. 创建 Loss
# =========================

loss_fn = nn.MSELoss()


# =========================
# 3. 创建 Adam
# =========================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# =========================
# 4. 准备数据
# =========================

x = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0]
])

y = torch.tensor([
    [3.0],
    [5.0],
    [7.0]
])


# =========================
# 5. 训练
# =========================

for epoch in range(100):

    # 模型预测
    prediction = model(x)

    # 计算 Loss
    loss = loss_fn(prediction, y)

    # 清除旧梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    if epoch % 10 == 0:
        print(
            "Epoch:",
            epoch,
            "Loss:",
            loss.item()
        )