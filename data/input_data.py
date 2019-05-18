"""
load data
split train set, test set.
"""
import pickle as pkl
import numpy as np
from feature import get_design_matrix
from sklearn.feature_selection import SelectKBest, chi2


def reduce_dim(feats, labels, dim):
    """
    使用卡方检验给属性降维度
    """
    selector = SelectKBest(chi2, k=dim)
    matrix = selector.fit_transform(feats, labels)

    mask = selector.get_support()
    with open('chi2_mask.pkl', 'wb') as f:
         pkl.dump(mask, f)

    return matrix


def getData(report_path, api_dict):
    """
    feats: 设计矩阵，每个样本表示为2000维的向量
    labels: 1表示malware, 2表示normal
    train:test = xx : 2000    2000个测试样本，其余都是训练样本
    """
    feats, labels = get_design_matrix(report_path, api_dict)

    feats= reduce_dim(feats, labels, 2000)

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
     report_path = "./reports"
     api_dict = './mapping_5.1.1.csv'
     
     train_x, train_y, test_x, test_y = getData(report_path, api_dict)
     print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
     data = dict()
     data['train_x'] = train_x
     data['train_y'] = train_y
     data['test_x'] = test_x
     data['test_y'] = test_y
     with open('data_set.pkl', 'wb') as f:
          pkl.dump(data, f)
