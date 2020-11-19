import numpy as np

import keras
from keras.layers import Conv2D, Add, GlobalAveragePooling2D, BatchNormalization, MaxPooling2D, ReLU, Flatten, Dense, Dropout, Concatenate, Softmax, AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.models import Model, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

class DenseNet:
    def __init__(self, classes=10):
        self.classes = classes
        self.model = self.build_model()

    def build_model(self, inputs_shape=(224, 224, 3), theta=0.5):
        inputs_cnn = Input(inputs_shape)

        x = Conv2D(6, (7, 7), strides=2)(inputs_cnn)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = self._dense_block(x, kernels=6)
        x = Conv2D(int(theta * 6 * 4), (1, 1))(x)
        x = AveragePooling2D((2, 2), strides=2)(x)

        x = self._dense_block(x, kernels=12)
        x = Conv2D(int(theta * 12 * 4), (1, 1))(x)
        x = AveragePooling2D((2, 2), strides=2)(x)

        x = self._dense_block(x, kernels=32)
        x = Conv2D(int(theta * 32 * 4), (1, 1))(x)
        x = AveragePooling2D((2, 2), strides=2)(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs_cnn, x)
        return model


    def _dense_unit(self, inputs, kernels=6):
        x1 = BatchNormalization()(inputs)
        x1 = ReLU()(x1)
        x1 = Conv2D(kernels, (1, 1))(x1)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = ZeroPadding2D(padding=(1, 1), dim_ordering='default')(x1)
        x1 = Conv2D(kernels, (3, 3))(x1)
        return x1

    def _dense_block(self, inputs, kernels=6):
        x1 = self._dense_unit(inputs, kernels)

        x2_input = Concatenate(axis=-1)([inputs, x1])
        x2 = self._dense_unit(x2_input, kernels)

        x3_input = Concatenate(axis=-1)([inputs, x1, x2])
        x3 = self._dense_unit(x3_input, kernels)

        x4_input = Concatenate(axis=-1)([inputs, x1, x2, x3])
        x4 = self._dense_unit(x4_input, kernels)

        output = Concatenate(axis=-1)([inputs, x1, x2, x3, x4])
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