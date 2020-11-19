import numpy as np

import keras
from keras.layers import Conv2D, MaxPooling2D, ReLU, Flatten, Dense, Dropout
from keras.models import Model, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

class VGG11:
    def __init__(self, inputs_shape):
        inputs = Input(shape=inputs_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        # x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(10, activation='softmax', kernel_regularizer=l2(5e-4))(x)

        self.model = Model(inputs, x)

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

class VGG13:
    def __init__(self,inputs_shape):
        inputs = Input(shape=inputs_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(10, activation='softmax', kernel_regularizer=l2(5e-4))(x)

        self.model = Model(inputs, x)

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


class VGG16_1:
    def __init__(self, inputs_shape):
        inputs = Input(shape=inputs_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(10, activation='softmax', kernel_regularizer=l2(5e-4))(x)

        self.model = Model(inputs, x)

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


class VGG16_2:
    def __init__(self, inputs_shape):
        inputs = Input(shape=inputs_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(10, activation='softmax', kernel_regularizer=l2(5e-4))(x)

        self.model = Model(inputs, x)

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


class VGG19:
    def __init__(self, inputs_shape):
        inputs = Input(shape=inputs_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (1, 1), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(5e-4))(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(4096, activation='relu', kernel_regularizer=l2(5e-4))(x)
        x = Dense(10, activation='softmax', kernel_regularizer=l2(5e-4))(x)

        self.model = Model(inputs, x)

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
