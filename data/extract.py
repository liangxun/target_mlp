import os
import sys
import json


class Extractor:
    def __init__(self, path_decompiled, api_dict):
        self.path_decompiled = path_decompiled
        self.sensitive_apis = self.getSensitiveAPIs(api_dict)
    
    def getSensitiveAPIs(self, api_dict):
        """建立敏感API字典"""
        APIs = set()
        with open(api_dict, 'r') as f:
            for line in f.readlines():
                CallerClass, CallerMehod = line.split(',')[:2]
                api = 'L' + CallerClass + ';->' + CallerMehod
                api = api.strip()
                # if api in APIs:
                    # print("Redundant api: {}".format(api))
                APIs.add(api)
        print("build dict: contain {} sensitive APIs.".format(len(APIs)))
        return APIs

    def analysis_smali(self, smali_path):
        for a, _, c in os.walk(smali_path):
            for file in c:
                single_smali_file = os.path.join(a,file)
                # print(single_smali_file)
                apis = self.extract_api(single_smali_file)
                all_apis = all_apis | apis
        return all_apis

    def extract_api(self, smali_file):
        apis = set()
        with open(smali_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line.startswith('invoke'):
                    func = line.split(' ')[-1]
                    func = func[:func.index('(')]
                    if func in self.sensitive_apis:
                        # print(func)
                        apis.add(func)
        return apis

    def extract(self, apk):
        smali_path = os.path.join(self.path_decompiled, apk, 'smali')
        apis = self.analysis_smali(smali_path)
        return list(apis)


if __name__ == '__main__':
    assert len(sys.argv) == 2
    tag = sys.argv[1] # tag="malware" or "normal"

    api_dict = '/home/security/Android/static/mapping_5.1.1.csv'
    path_decompiled = '/home/security/data/decompiled/{}'.format(tag)
    apks_path = "/home/security/data/{}_apks".format(tag) # 遍历时用到最原始的apk目录，没有实际作用。因为反编译和提取第三方库都存在解析失败的APK,所以只有原始apk文件中的是全集
    
    out_path = "./reports/{}".format(tag)

    E = Extractor(path_decompiled, api_dict)
 
    cnt = 1
    error_cnt = 0
    for apk in os.listdir(apks_path):
        if os.path.exists(os.path.join(out_path, apk)):
            print("{}, {} already exists".format(cnt, apk))
        else:
            try:
                print("{}, {}".format(cnt, apk))
                apis = E.extract(apk)
                report_file = os.path.join(out_path, apk)
                with open(report_file, 'w') as f:
                    json.dump(apis, f)
            except Exception as e:
                print("{}, error \n{}".format(apk,e))
                error_cnt += 1
                pass
        cnt += 1
