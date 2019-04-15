"""
预测单个样本实例
"""
from keras.models import load_model
import numpy as np
from input_data import getData


train_data, train_label, test_data, test_label = getData()
print(test_data.shape, test_label.shape)
model = load_model('target_model')

index = np.random.randint(len(test_label))
sample = test_data[index][np.newaxis, :]
sample_true = int(test_label[index])
print(index)
print(sample.shape)

sample_pred = model.predict(sample)
sample_pred = np.argmax(sample_pred.squeeze())

print(sample_pred, sample_true)
