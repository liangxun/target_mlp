# target_mlp
简单的恶意软件检测模型，作为被攻击对象  

**特征说明**：每个apk表示为2000-dim的向量；特征中仅包括敏感api，即向量的每一维度代表apk是否调用某个api（从pscout中筛选的2000个api）   
**标签说明**：1表示恶意软件，0表示正常软件。

## 任务
推测特征中使用的是哪2000个api？

## 测试单个APK
run:
``` 
python3 example.py malware ff9421ecebce5662b8ca8d1c11547032.apk
```
