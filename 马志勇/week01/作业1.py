import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

test_query=input('请输入您要预测的内容:')

dataset = pd.read_csv(filepath_or_buffer=r"D:\nlp-epda\hub-ktxw\马志勇\week01\dataset.csv", sep="\t", header=None)
input_sententce = dataset[0].apply(lambda x : "".join(jieba.lcut(x)))

def print_head(title: str) -> None:
    format_str = "-" * 20 + title + "-" * 20
    print(format_str)


def case_01():
    case_title = "KNN模型 预测"
    print_head(case_title)

    vector = TfidfVectorizer()
    vector.fit(input_sententce.values)
    input_feature = vector.transform(input_sententce.values)

    knn_model = KNeighborsClassifier()
    knn_model.fit(input_feature,dataset[1].values)

    test_sentence = "".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])

    print("待预测的文本", test_query)
    print(f"{case_title} 结果:", knn_model.predict(test_feature))


def case_02():
    case_title = "线性模型 预测"
    print_head(case_title)

    vector = TfidfVectorizer()
    vector.fit(input_sententce.values)
    input_feature = vector.transform(input_sententce.values)

    lm = LogisticRegression()
    lm.fit(input_feature, dataset[1].values)

    test_sentence = "".join(jieba.lcut(test_query))
    test_feature = vector.transform([test_sentence])

    print("待预测的文本", test_query)
    print(f"{case_title} 结果:", lm.predict(test_feature))

case_01()
case_02()