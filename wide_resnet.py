from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import regularizers


def build_basic_block(x, base_width=0, N=0, k=0, dropout=0.3, strides=(1, 1), weight_decay=0.0005):
    x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform',
                           kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)


    z = Conv2D(base_width * k, (3, 3,), padding="same", kernel_initializer="he_normal", use_bias=False,
               kernel_regularizer=regularizers.l2(weight_decay))(x)



