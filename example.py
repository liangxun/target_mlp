"""
预测单个样本实例, 输入为apk（samli)
"""
import os
import sys
from keras.models import load_model
import numpy as np
import pickle as pkl
from data.extract import Extractor
from data.feature import build_api2index


path_decompiled = '/home/security/data/decompiled'
api_dict = '/home/security/Android/static/mapping_5.1.1.csv'
chi2_path = "/home/security/target_mlp/data/chi2_selector"

api2index = build_api2index(api_dict)
with open(chi2_path, 'rb') as f:
    chi2 = pkl.load(f)


def get_sample(tag, apk):    
    # 提取apis
    path = os.path.join(path_decompiled, tag)
    E = Extractor(path, api_dict)
    apis = E.extract(apk)

    # 将apis调用转换成向量
    feat = np.zeros(len(api2index))
    for api in apis:
        feat[api2index[api]] = 1
    
    # 降维
    feat = chi2.transform(feat)

    # 获取标签
    if tag == 'malware':
        label = 1
    elif tag == 'normal':
        label = 0
    else:
        print("error")
    
    return feat, label


def run(tag, apk):
    x, y_true = get_sample(tag, apk)
    model = load_model('target_model')
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred.squeeze())
    print('y_true={}\ty_pred={}.'.format(y_true, y_pred))
    return y_true, y_pred


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('error')
    tag = sys.argv[1] # tag = 'malware' or 'normal'
    apk = sys.argv[2]

    run(tag, apk)
