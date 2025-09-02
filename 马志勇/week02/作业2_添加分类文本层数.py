import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
"""
注意:该case包含了 深度学习的所有过程
"""

# 通过读取原始文本(第一列-文档 和 第二列-类别),构建 字 与 数字编码 的映射 字典
def build_char_and_num_dics():
    """
    通过读取原始文本(第一列-文档 和 第二列-类别),构建 字 与 数字编码 的映射 字典
    :return:
        texts, string_labels, char_to_index, index_to_char, label_to_index,numerical_labels
    """
    dataset = pd.read_csv("../data/dataset.csv", sep="\t", header=None)

    # 获取数据集 的 第一列(文本)
    texts = dataset[0].tolist()

    # 获取数据集 的 第二列(类别、标签)
    string_labels = dataset[1].tolist()

    # 类别 转换为 数字
    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    numerical_labels = [label_to_index[label] for label in string_labels]
    print(f"label_to_index: {label_to_index}")
    print(f"numerical_labels: {numerical_labels}")

    # 原始文本 - 将文本 构建为一个 词典, 把字 编码成数字, key = 字 , value = 数字编码
    # 构建 字 -> 数字编码 的字典
    char_to_index = {'<pad>':0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    # 构建 数字编码 -> 字 的字典
    index_to_char = {i: char for char, i in char_to_index.items()}
    return texts, string_labels, char_to_index, index_to_char, label_to_index, numerical_labels

# 构建 词频向量 数据集 (同时截断过长的文本)
class CharBowDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    # 截断文档
    def _truncate(self):
        """
        截断文档
        :param texts:
        :param char_to_index:
        :param max_len:
        :return:
        """
        # 获取每个 文本的 前40个字符(截断 超过40的字符)

        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)
        return tokenized_texts

    # 构建 term frequency - 词频向量 数据集
    # 注意, 稀疏矩阵也会导致过拟合,因此 使用 embeeding(稠密输出)层 代替 term frequency
    def _create_bow_vectors(self):
        """
        构建 term frequency - 词频向量 数据集
        注意, 稀疏矩阵也会导致过拟合,因此 使用 embeeding(稠密输出)层 代替 term frequency
        :param tokenized_texts:
        :param vocab_size:
        :return:
        """
        bow_vectors = []

        tokenized_texts = self._truncate()

        for text_indices in tokenized_texts:
            # 词典个数长度的向量, 存储每个字在这个文本中出现的次数
            bow_vector = torch.zeros(self.vocab_size)

            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 这里 __init__ 函数 做层初始化, 隐藏层的个数 和 验证精度

        # 作业 层数和节点个数，对比模型的loss变化
        super(SimpleClassifier,self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()


        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(hidden_dim, hidden_dim*2)
        self.relu5 = nn.ReLU()

        self.fc6 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu6 = nn.ReLU()

        self.fc7 = nn.Linear(hidden_dim, hidden_dim*2)
        self.relu7 = nn.ReLU()

        self.fc8 = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = self.relu3(out)

        out = self.fc4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        out = self.relu5(out)

        out = self.fc6(out)
        out = self.relu6(out)

        out = self.fc7(out)
        out = self.relu7(out)

        out = self.fc8(out)

        return out


def build_model_lossfunc_optim(output_dim, vocab_size, hidden_dim=128):
    """
    定义模型、损失函数和优化器
    """
    model = SimpleClassifier(vocab_size, hidden_dim, output_dim)

    # 交叉商损失函数, 属于分类损失函数 用于衡量模型输出的结果和类别对应的关系
    loss_func = nn.CrossEntropyLoss()

    # 这里使用 Adam优化器 可以结合梯度 动态调整学习率 lr
    # lr(学习率) 太大会造成梯度爆炸, 太小会到达不了 局部最优
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    return model, loss_func, optimizer



def training(dataloader, model, criterion, optimizer, num_epochs=10):
    """
    训练模型
    :param dataloader:
    :param model:
    :param optimizer:
    :param criterion:
    :param training_num:
    :return:
    """
    # 循环比遍历10次 训练和更新模型
    # 训练和更新后 就可以为给定的文本进行推理
    # 推理其实就是做正向传播,把我们传入的文本也转变成向量,然后通过模型去推理
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 20 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 分类文本
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    分类文本
    :param text:
    :param model:
    :param char_to_index:
    :param vocab_size:
    :param max_len:
    :param index_to_label:
    :return:
    """
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    # 正向传播, 11个神经元的输出
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    # 选择一下 11个神经元 哪个 知信度 比较高,如果哪个位置比较高, 就任务对应哪个位置
    _, predicted_index = torch.max(output,1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]
    return predicted_label

if __name__ == "__main__":
    max_len = 40
    # step -1 通过读取原始文本(第一列-文档 和 第二列-类别),构建 字 与 数字编码 的映射 字典
    texts, string_labels, char_to_index, index_to_char, label_to_index, numerical_labels = build_char_and_num_dics()
    index_to_label = {i: label for label, i in label_to_index.items()}

    vocab_size = len(char_to_index)
    output_size = len(label_to_index)
    # 读取单个样本 - 构建 词频向量 数据集 (同时截断过长的文本)
    char_dataset = CharBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size=vocab_size)

    # 读取批量数据集 即 batch数据
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

    # step -3 定义模型、损失函数以及优化器
    output_dim = len(label_to_index)
    model, criterion, optimizer = build_model_lossfunc_optim(output_dim, vocab_size)

    # step -4 training model
    training(dataloader, model, criterion, optimizer, num_epochs=10)



    new_text = "帮我导航到北京"
    predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")