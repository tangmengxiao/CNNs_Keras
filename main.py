import numpy as np

import keras
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from utils import load_data_mnist
from VGG import VGG11, VGG13, VGG16_1, VGG16_2, VGG19
from GoogLeNet import GoogLeNet
from ResNet import ResNet
from DenseNet import DenseNet

np.random.seed()
# 加载数据
train_data, train_labels, test_data, test_labels = load_data_mnist('data/mnist/MNIST_data.npz')

# 归一化
train_data = train_data / 255
test_data = test_data / 255

# 打乱
idx = np.arange(np.shape(train_labels)[0])
np.random.shuffle(idx)
train_data = train_data[idx]

idx = np.arange(np.shape(test_labels)[0])
np.random.shuffle(idx)
test_data = test_data[idx]

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

# 建立网络
#model_VGG = VGG11(inputs_shape=np.shape(train_data[0]))
#model_VGG.get_model().summary()

model = DenseNet()
model.get_model().summary()

# 训练
opti = SGD(lr=1e-2, momentum=0.9)
myEarlyStoping = EarlyStopping(monitor='val_loss', patience=20)
model_VGG.train(opti=opti, loss='categorical_crossentropy',
                x=train_data, y=train_labels, batch_size=256, epochs=500,
                callbacks=[myEarlyStoping], validation_split=0.1)

# 模型评估
scores = model_VGG.evaluate(x=test_data, y=test_labels)


