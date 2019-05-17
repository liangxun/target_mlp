"""
预测单个样本实例, 输入为apk（samli)
"""
import os
import sys
from keras.models import load_model
import numpy as np
import pickle as pkl
from data.extract import Extractor
import json
from data.feature import build_api2index


report_path = './data/reports'
api_dict = './data/api2index.pkl'
mask_path = './data/chi2_mask.pkl'
model_path = 'target_model.keras'

with open(mask_path, 'rb') as f:
    chi2_mask = pkl.load(f)
with open(api_dict, 'rb') as f:
    api2index = pkl.load(f)
model = load_model(model_path)


def get_sample(tag, apk):    
    # 提取apis
    path = os.path.join(report_path, tag)
    file = os.path.join(path, apk)
    with(open(file, 'r')) as f:
        apis = json.load(f)

    # 将apis调用转换成向量
    feat = np.zeros(len(api2index))
    for api in apis:
        feat[api2index[api]] = 1
    
    # 降维
    feat = feat[chi2_mask]

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
    print(x.shape)
    y_pred = model.predict(x[np.newaxis, :])
    y_pred = np.argmax(y_pred.squeeze())
    return y_true, y_pred


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('error')
    tag = sys.argv[1] # tag = 'malware' or 'normal'
    apk = sys.argv[2]

    y_true, y_pred = run(tag, apk)
    print("apk: {}".format(apk))
    print('y_true={}\ty_pred={}.'.format(y_true, y_pred))
