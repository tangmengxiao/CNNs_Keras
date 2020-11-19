import numpy as np

import keras
from keras.layers import Conv2D, MaxPooling2D, ReLU, Flatten, Dense, Dropout, Concatenate, Softmax, AveragePooling2D
from keras.models import Model, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2


class GoogLeNet:
    def __init__(self, classes=10):
        self.classes = classes
        self.model = self.build_model()

    def build_model(self, inputs_shape=(224, 224, 3)):
        input_googlenet = Input(shape=inputs_shape)

        x = Conv2D(64, (7, 7), strides=2, activation='relu')(input_googlenet)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(64, (1, 1), activation='relu')(x)
        x = Conv2D(192, (3, 3), activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = self._inception(inputs=x, kernels_11=64,
                            kernels_33_reduce=96, kernels_33=128,
                            kernels_55_reduce=16, kernels_55=32, kernels_poolproj=32, id='3a')
        x = self._inception(inputs=x, kernels_11=128,
                            kernels_33_reduce=128, kernels_33=192,
                            kernels_55_reduce=32, kernels_55=96, kernels_poolproj=64, id='3b')
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = self._inception(inputs=x, kernels_11=192,
                            kernels_33_reduce=96, kernels_33=208,
                            kernels_55_reduce=16, kernels_55=48, kernels_poolproj=64, id='4a')
        x, bybass_0 = self._inception(inputs=x, kernels_11=160,
                                      kernels_33_reduce=112, kernels_33=224,
                                      kernels_55_reduce=24, kernels_55=64, kernels_poolproj=64, id='4b',
                                      with_softmax=True)
        x = self._inception(inputs=x, kernels_11=128,
                                      kernels_33_reduce=128, kernels_33=256,
                                      kernels_55_reduce=24, kernels_55=64, kernels_poolproj=64, id='4c',
                                      with_softmax=False)
        x = self._inception(inputs=x, kernels_11=112,
                                      kernels_33_reduce=144, kernels_33=288,
                                      kernels_55_reduce=32, kernels_55=64, kernels_poolproj=64, id='4d',
                                      with_softmax=False)
        x, bybass_1 = self._inception(inputs=x, kernels_11=256,
                                      kernels_33_reduce=160, kernels_33=320,
                                      kernels_55_reduce=32, kernels_55=128, kernels_poolproj=128, id='4e',
                                      with_softmax=True)
        x = MaxPooling2D((3, 3), strides=2)(x)

        x = self._inception(inputs=x, kernels_11=256,
                            kernels_33_reduce=160, kernels_33=320,
                            kernels_55_reduce=32, kernels_55=128, kernels_poolproj=128, id='5a',
                            with_softmax=False)
        x = self._inception(inputs=x, kernels_11=384,
                            kernels_33_reduce=192, kernels_33=384,
                            kernels_55_reduce=48, kernels_55=128, kernels_poolproj=128, id='5b',
                            with_softmax=False)

        x = AveragePooling2D((5, 5), strides=1)(x)
        x = Dropout(0.4)(x)
        x = Flatten()(x)

        x = Dense(self.classes, activation='softmax')(x)

        model = Model(input_googlenet, [x, bybass_0, bybass_1])
        return model


    def _inception(self, inputs, kernels_11=64, kernels_33_reduce=96, kernels_33=128, kernels_55_reduce=16, kernels_55=32, kernels_poolproj=32, with_softmax=False, id='3a'):
        pass1 = Conv2D(kernels_11, (1, 1), padding='same', activation='relu', name='Conv11_'+id)(inputs)

        pass2 = Conv2D(kernels_33_reduce, (1, 1), padding='same', activation='relu', name='Conv33_reduce_'+id)(inputs)
        pass2 = Conv2D(kernels_33, (3, 3), padding='same', activation='relu', name='Conv33_'+id)(pass2)

        pass3 = Conv2D(kernels_55_reduce, (1, 1), padding='same', activation='relu', name='Conv55_reduce_'+id)(inputs)
        pass3 = Conv2D(kernels_55, (5, 5), padding='same', activation='relu', name='Conv55_'+id)(pass3)

        pass4 = MaxPooling2D((3, 3), strides=1, padding='same')(inputs)
        pass4 = Conv2D(kernels_poolproj, (1, 1), padding='same',activation='relu', name='Conv_poolproj_'+id)(pass4)

        output = Concatenate(axis=-1, name='Concate_'+id)([pass1, pass2, pass3, pass4])

        if with_softmax:
            out_pass = AveragePooling2D((5, 5), strides=3)(inputs)
            out_pass = Conv2D(128, (1, 1), activation='relu', name='outpath_Conv11_'+id)(out_pass)
            out_pass = Dense(1024, activation='relu', name='outpath_Dense_'+id)(out_pass)
            out_pass = Dropout(0.7)(out_pass)
            out_pass = Dense(self.classes, activation='softmax', name='outpath_output'+id)(out_pass)
            return output, out_pass
        else:
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