"""
load data
split train set, test set.
"""
import pickle as pkl
import numpy as np
from feature import get_design_matrix
from sklearn.feature_selection import SelectKBest, chi2


def reduce_dim(feats, labels, dim,  save_selector):
    """
    使用卡方检验给属性降维度
    """
    selector = SelectKBest(chi2, k=dim)
    matrix = selector.fit_transform(feats, labels)

    # 保存降维器，给单个apk降维时还会用到
    with open(save_selector, 'rb') as f:
         pkl.dump(selector, f)

    return matrix


def getData(report_path, api_dict, save_selector):
    """
    feats: 设计矩阵，每个样本表示为2000维的向量
    labels: 1表示malware, 2表示normal
    selector: 卡方检验降维度器
    train:test = xx : 2000    2000个测试样本，其余都是训练样本
    """
    feats, labels = get_design_matrix(report_path, api_dict)

    feats= reduce_dim(feats, labels, 2000, save_selector)

    X = feats.toarray()
    y = labels[:, np.newaxis]

    data = np.hstack((X, y))
    malware = data[:np.sum(y==1)]
    benign = data[np.sum(y==1):]
    np.random.shuffle(malware)
    np.random.shuffle(benign)

    train = np.vstack((malware[:-1000], benign[:-1000]))
    test = np.vstack((malware[-1000:], benign[-1000:]))
    np.random.shuffle(train)
    np.random.shuffle(test)

    train_data = train[:, :-1]
    train_label = train[:, -1]
    test_data = test[:, :-1]
    test_label = test[:, -1]
    return train_data, train_label, test_data, test_label


if __name__ == '__main__':
     report_path = "/home/security/data/reports"
     api_dict = '/home/security/Android/static/mapping_5.1.1.csv'
     save_selector = './chi2_selector.pkl'
     train_x, train_y, test_x, test_y = getData(report_path, api_dict, save_selector)
     print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
