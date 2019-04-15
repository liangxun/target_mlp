"""
load data
split train set, test set.
"""
import pickle as pkl
import numpy as np


def getData(data_path='features.pkl'):
    with open(data_path,'rb') as f:
        data = pkl.load(f)
    X = data['feat_2000']   # feat_2000, feat_3000
    y = data['labels']
    X = X.toarray()
    y = y[:, np.newaxis]

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
     train_x, train_y, test_x, test_y = getData()
     print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
