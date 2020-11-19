import numpy as np

import keras
from keras.layers import Conv2D, Add, BatchNormalization, MaxPooling2D, ReLU, Flatten, Dense, Dropout, Concatenate, Softmax, AveragePooling2D
from keras.models import Model, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


class ResNet:
    def __init__(self, classes=10):
        self.classes = classes
        self.model = self.build_model()

    def build_model(self, inputs_shape=(224, 224, 3)):
        inputs_cnn = Input(inputs_shape)

        x = Conv2D(64, (7, 7), strides=2)(inputs_cnn)
        x = MaxPooling2D((2, 2), strides=2)(x)

        x = self._resblock(x, kernels=64)
        x = self._resblock(x, kernels=64)
        x = self._resblock(x, kernels=64)

        x = self._resblock(x, kernels=128, strides=1)
        x = self._resblock(x, kernels=128)
        x = self._resblock(x, kernels=128)
        x = self._resblock(x, kernels=128)

        x = self._resblock(x, kernels=256, strides=1)
        x = self._resblock(x, kernels=256)
        x = self._resblock(x, kernels=256)
        x = self._resblock(x, kernels=256)
        x = self._resblock(x, kernels=256)
        x = self._resblock(x, kernels=256)

        x = self._resblock(x, kernels=512, strides=1)
        x = self._resblock(x, kernels=512)
        x = self._resblock(x, kernels=512)

        x = Flatten()(x)
        x = Dense(self.classes, activation='softmax')(x)
        model = Model(inputs_cnn, x)
        return model

    def _resblock(self, inputs, kernels=64, strides=1):
        x0 = Conv2D(kernels, (1, 1), padding='same')(inputs)
        x0 = BatchNormalization()(x0)
        x0 = ReLU()(x0)

        x1 = Conv2D(kernels, (3, 3), padding='same', strides=strides)(inputs)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)

        x2 = Conv2D(kernels, (3, 3), padding='same')(x1)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        output = Add()([x0, x1, x2])
        return output

    def get_model(self):
        return self.model

    def train(self, opti, loss, x, y, batch_size, epochs, callbacks, validation_split):
        self.model.compile(optimizer=opti,
                      loss=loss,
                      metrics=['accuracy'])
        hist = self.model.fit(x=x,
                  y=y,
                  batch_size=batch_size,
                  epochs=epochs,
                  callbacks=callbacks,
                  validation_split=validation_split)
        return hist

    def evaluate(self, x, y):
        scores = self.model.evaluate(x=x, y=y)
        print('acc', scores[1])
        return scores