"""
静态特征提取阶段为每个apk生成了一个report报告，
Extractor的工作是将report转换成向量形式
"""
import os
import sys
import json
import pickle as pkl
import numpy as np
from scipy import sparse
from scipy.sparse import vstack


def build_api2index(api_dict):
    """建立敏感API字典"""
    APIs = set()
    with open(api_dict, 'r') as f:
        for line in f.readlines():
            CallerClass, CallerMehod = line.split(',')[:2]
            api = 'L' + CallerClass + ';->' + CallerMehod
            api = api.strip()
            APIs.add(api)

    api2index = dict()
    for i, j in enumerate(APIs):
        api2index[j] = i
    return api2index


def build_apk2index(apks_path):
    apks = os.listdir(apks_path)
    dic = dict()
    for i, apk in enumerate(apks):
        dic[apk] = i
    return dic


class Extractor:
    def __init__(self, report_path, api_dict):
        self.report_path = report_path

        self.api2index = build_api2index(api_dict)
        self.apk2index = build_apk2index(self.report_path)

        self.feat = None
    
    def feature_matrix(self):
        """
        构造特征矩阵
        :return: features.shape = (apk, api)
        """
        print('contain {} apks'.format(len(self.apk2index)))
        print('contain {} apis'.format(len(self.api2index)))
        print("each apk is represented by {}-dim binary vector".format(len(self.api2index)))
        
        feat = np.zeros((len(self.apk2index), len(self.api2index)))

        cnt = 1
        for apk in self.apk2index.keys():
            # print(cnt, apk)
            cnt += 1

            file = os.path.join(self.report_path, apk)
            with(open(file, 'r')) as f:
                report = json.load(f)
            apis = report['sensitive_api']
            for api in apis:
                # feat[self.apk2index[apk]][self.api2index[api]] += 1    # 统计调用频次
                feat[self.apk2index[apk]][self.api2index[api]] = 1
        feat = sparse.csr_matrix(feat)
        self.feat = feat
    
    def run(self):
        self.feature_matrix()
        ret = self.feat
        return ret


def get_design_matrix(reports, api_dict):
    print("handle malware...")
    E = Extractor(os.path.join(reports, 'malware'), api_dict)
    malware = E.run()
    print("handle normal...")
    E = Extractor(os.path.join(reports, 'normal'), api_dict)
    normal = E.run()

    apk_api = vstack((malware['feature_matrix'], normal['feature_matrix']))

    num_malware = malware['feature_matrix'].shape[0]
    num_normal = normal['feature_matrix'].shape[0]
    labels = np.array([1]*num_malware + [0]*num_normal) # 1: malware; 0: bengin/normal
    print("total {} apks in dataset. \n\tnum_malware={}\n\tnum_normal={}".format(len(labels), num_malware, num_normal))
    
    return apk_api, labels


if __name__ == '__main__':
    report_path = "/home/security/data/reports"
    api_dict = '/home/security/Android/static/mapping_5.1.1.csv'
    a, b = get_design_matrix(report_path, api_dict)
    print(a.shape, len(b))
