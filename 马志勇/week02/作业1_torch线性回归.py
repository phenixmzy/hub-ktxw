import torch
import numpy as np # CPU环境(非深度学习中)下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 0 构建数据集合, 并展示图
# def test_build_dataset():
#     test_X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
#     test_y_numpy = np.sin(test_X_numpy)
#
#     # 将 NumPy数组转换为PyTorch 张量
#     test_X = torch.from_numpy(test_X_numpy).float()
#     test_y = torch.from_numpy(test_y_numpy).float()
#
#     test_model = nn.Linear(1,1)
#     test_loss_fn = torch.nn.MSELoss()  # 回归任务里面的损失函数
#     test_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.01)
#     test_num_epochs = 5000
#     for epoch in range(test_num_epochs):
#         test_y_pred = test_model(test_X)
#         test_loss = test_loss_fn(test_y_pred, test_y)
#         test_optimizer.zero_grad()
#         test_loss.backward()
#         test_optimizer.step()
#
#         if (epoch + 1) % 100 == 0:
#             print(f'Epoch [{epoch + 1}/{test_num_epochs}], Loss: {test_loss.item():.4f}')
#
#     test_model.eval()
#     with torch.no_grad():
#         test_y_predicted = test_model(test_X).numpy()
#
#     plt.figure(figsize=(10, 6))
#     plt.scatter(test_X_numpy, test_y_numpy, label='test Raw data', color='blue', alpha=0.6)
#     plt.plot(test_X_numpy, test_y_predicted, label=f'test_Model', color='red', linewidth=2)
#     plt.xlabel('X')
#     plt.ylabel('y')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# test_build_dataset()


# step-1. 生成模拟数据 (与之前相同)
## 形状为(100, 1) 左闭右开的 二维数组, 其中包括 100个在[0,1) 范围内均匀分布的随机浮点数
# X_numpy = np.random.rand(100, 1) * 10
# 批量训练 100 * 1
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

## y的 np.random.randn(100, 1) 是一个扰动噪音,是演示中故意增加难度
#y_numpy = 2 * X_numpy + 1 + np.random.randn(100, 1)
y_numpy = np.sin(X_numpy)
#+ np.random.randn(100, 1)/20

## 从numpy 转成 tensor 张量, torch中 所有的计算通过tensor计算
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# step-3. 定义 多层的前馈网络模型-全连接网络、LossFunction损失函数 和 优化器
# 定义模型 - 层数越多, 模型复杂度越高 拟合能力越强
model = nn.Sequential(
    # 随机初始化
    nn.Linear(1, 10),
    nn.ReLU(),
    # nn.Dropout(0.5),

    nn.Linear(10,10),
    nn.ReLU(),

    nn.Linear(10, 10),
    nn.ReLU(),

    nn.Linear(10, 10),
    nn.ReLU(),

    nn.Linear(10, 10),
    nn.ReLU(),

    nn.Linear(10, 10),
    nn.ReLU(),

    nn.Linear(10, 1)
)

# 损失函数仍然是均方误差 (MSE)。
# 定义损失函数, 回归任务 衡量 a * x + b < - > y"
loss_fn = torch.nn.MSELoss() # 回归任务里面的损失函数

# 优化器 定义
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 核心 - 训练模型 通过循环的方式不断的更新模型的参数
# num_epochs 迭代1000次
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    # y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度, torch 梯度是累加的,因此要清空梯度,不然会对效果有影响
    loss.backward()        # model 自动 计算梯度
    optimizer.step()       # model 自动 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
# a_learned = a.item()
# b_learned = b.item()
# print(f"拟合的斜率 a: {a_learned:.4f}")
# print(f"拟合的截距 b: {b_learned:.4f}")
# print("---" * 10)

# 将模型切换到评估模式, 强烈建议训练结束后 执行的步骤, 这是一个非常好的习惯
# 主动关闭dropout
model.eval()

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值

with torch.no_grad():
    # 使用训练好的模型进行预测
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'Model', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
