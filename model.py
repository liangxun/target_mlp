import time
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from data.input_data import getData
from metrics import Metrics
import pickle as pkl

# =================== configure =========================
epochs = 10
batchsize = 64
lr = 0.05
dropout_r = 0.1

# ===================load data========================
data_path = "./data/data_set.pkl"
with open(data_path, 'rb') as f:
    data = pkl.load(f)
train_data = data['train_x']
train_label=data['train_y']
test_data = data['test_x']
test_label = data['test_y']
train_label = to_categorical(train_label)

# ================== define model ================================
def bpnn(input_dim, layers_out, lr=0.001, dropout=0.5):
    model = Sequential()
    model.add(Dense(layers_out[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout))
    model.add(Dense(layers_out[1], activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(layers_out[2], activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

input_dim = train_data.shape[1]
layers_out = [256, 128, 2]
model = bpnn(input_dim, layers_out, lr=lr, dropout=dropout_r)
print(model.summary())

# =================== training =====================================
start_time = time.time()
hist = model.fit(train_data,
                 train_label,
                 batch_size=batchsize,
                 epochs=epochs)
print('Training duration: %d(s).' % (time.time()-start_time))
model.save("target_model")


def plot_train(history):
    """打印出训练过程中loss的变化趋势"""
    train_loss = history['loss']
    train_acc = history['acc']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(train_loss, 'g-', label='train_loss')
    ax2 = ax1.twinx()
    line2 = ax2.plot(train_acc, 'r', label='train_acc')

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='best')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('train_accuracy')
    plt.savefig('train_process.png')

    plt.show()

plot_train(hist.history)

# ===================== predict and  analysis ========================
prediction = model.predict(test_data)
M = Metrics(test_label, prediction)
M.describe()