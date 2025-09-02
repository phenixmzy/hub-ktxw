import torch
import torch.nn as nn
# CPU环境(非深度学习中)下的句子运算和向量运算
import numpy as np

import matplotlib.pyplot as plt

import case.课程练习.utils.util as u
def print_head(head_name: str):
    u.head_str(head_name)

# 1. 生成模拟数据
def builder_simulated_data():
    print_head("1.开始生成模拟数据")

    # 形状为 (100,1) 左闭右开 的二维数组, 其中包括100个在[0,1) 范围内均匀分布的随机浮点数
    X_numpy = np.random.rand(100,1) * 10

    # y 的 np.random.rand(100,1) 是一个 扰动噪音, 是演示中故意增加难度
    y_numpy = 2 * X_numpy + np.random.rand(100,1)

    # 从 numpy 转成 tensor格式, torch中 所有的计算通过tensor计算
    X = torch.from_numpy(X_numpy).float()
    y = torch.from_numpy(y_numpy).float()
    print_head("1.模拟数据生产完成")
    return X_numpy, y_numpy, X, y

# 2. 直接创建参数张量 a 和 b(张量 在数学和物理学中的标准英文翻译是 tensor)
def create_tensor():
    print_head("2.生成张量 (即 tensor)")
    # 这里是主要修改的部分,我们不再使用 nn.Linear, torch.randn() 是生成随机值作为初始值
    # y = a * x + b
    # requires_grad=True 是关键,它告诉PyTorch 我们需要计算这些张量 的梯度
    a = torch.randn(1, requires_grad=True, dtype=torch.float)
    b = torch.randn(1, requires_grad=True, dtype=torch.float)
    print(f"初始参数 a: {a.item():.4f}")
    print(f"初始参数 b: {b.item():.4f}")
    return a,b

# 3. 定义LossFunction 损失函数和优化器
def create_lossfunc_and_optim(a,b, lr_value = 0.01):
    print_head("3. 定义LossFunction 损失函数和优化器")
    # 定义回归任务里面损失函数
    # 损失函数仍然是 均方误差(MSE), 定义一个损失函数,回归任务 衡量 a * x +b <-> Y"
    loss_fn = nn.MSELoss()

    # 优化器现在直接传入 我们手动创建的 参数a, b
    # Pytorch 会自动根据这些参数的梯度来更新它们
    # 优化器 基于 a, b 梯度自动更新
    # lr 表示学习率:
    # - 如果lr 设置比较合适, 能够经过几次迭代就能接近局部最优, loss值 趋于稳定
    # - 如果设置太大, 会出现梯度爆炸, loss 就会显示出变化大 不稳定, 不如出现 loss = Nan
    # - 如果设置太小, loss 会变化缓慢
    optimizer = torch.optim.SGD([a,b], lr = lr_value)
    return loss_fn, optimizer

# 4. 训练模型
# 训练模型通常是 通过循环的方式不断更新模型参数 num_epochs 迭代 1000次
def training_model(num_epochs=1000):

    X_numpy, y_numpy, X, y = builder_simulated_data()
    a, b = create_tensor()
    loss_fn, optimizer = create_lossfunc_and_optim(a,b)

    print_head("\n4.训练开始")
    for epoch in range(num_epochs):
        #4.1 前向传播(手动计算):  y_pred = a * X + b
        y_pred = a * X + b

        #4.2 计算损失
        loss = loss_fn(y_pred, y)

        #4.3 反向传播和优化
        ## 4.3.1 清空梯度, 因为torch梯度是累加的,因此要清空梯度,否则会对效果有影响
        optimizer.zero_grad()

        ## 4.3.2 计算梯度, 得到 a 和 b 的梯度
        loss.backward()

        ## 4.3.3 更新参数, 更新 a 和 b 的取值
        optimizer.step()

        # 每100个 epoch 打印一次
        if (epoch + 1) % 100 == 0:
            print(f"Epoch[{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print_head("\n4.训练完成")

    a_learned = a.item()
    b_learned = b.item()

    print_head("5.打印最终学到的参数")
    print(f"拟合的斜率 a: {a_learned:.4f}")
    print(f"拟合的截距 b: {b_learned:.4f}")

    print_head("6.绘制结果, 使用最终学习到的参数a 和 b 来计算拟合直线的y值")
    with torch.no_grad():
        y_predicted = a_learned * X + b_learned

    plt.figure(figsize=(10, 6))
    plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
    plt.plot(X_numpy, y_predicted, label=f'Model: y = {a_learned:.2f}x + {b_learned:.2f}', color='red', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    training_model(num_epochs=1000)


