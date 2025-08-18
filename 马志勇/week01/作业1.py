import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def print_head(title: str) -> None:
    format_str = "-" * 20 + title + "-" * 20
    print(format_str)


def run():
    print_head("1. 加载文本数据 和 预处理")
    dataset = pd.read_csv(filepath_or_buffer=r"../week01/dataset.csv", sep="\t", header=None)
    print("预览加载文本数据")
    print(dataset.head(5))

    print("jiaba进行中文分词, 并用空格进行分隔")
    input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
    labels = dataset[1]

    print_head("2. 提取特征向量")
    vectorizer = TfidfVectorizer() # CountVectorizer
    vectorizer.fit(input_sentences.values)
    input_features = vectorizer.transform(input_sentences.values)
    # input_features = vectorizer.fit_transform(input_sentences.values)


    print(f"打印 特征矩阵的形状: {input_features.shape}")

    print_head("3. 模型训练与评估")
    # 定义四种不同的模型
    models = {
        'KNN (K-Nearest Neighbors)': KNeighborsClassifier(),
        'Naive Bayes (MultinomialNB)': MultinomialNB(),
        'SVM (Support Vector Classifier)': SVC(kernel='linear'),
        'Decision Tree': DecisionTreeClassifier()
    }

    for name, model in models.items():
        print("\n")
        print_head(f"正在评估 {name} 模型")

        # 使用 cross_val_predict 进行 5次 交叉验证,获取每个样本的预测结果
        y_pred = cross_val_predict(model, input_features, labels, cv=5)


        # 基于交叉验证的预测结果, 计算并打印准确率和分类报告
        print(f"平均准确率: {accuracy_score(labels, y_pred):.4f}")
        report = classification_report(labels, y_pred, zero_division=0)
        print(report)



if __name__ == "__main__":
    run()