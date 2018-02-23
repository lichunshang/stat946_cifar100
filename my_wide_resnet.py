from keras.layers import *
from keras.models import Model
from keras.regularizers import l2
from keras import backend

_CHANNEL_AXIS = 1 if backend.image_data_format() == "channels_first" else -1

bn_weight_decay = 0.0005
cnn_weight_decay = 0.0005
fc_weight_decay = 0.0005


def _build_main_block(x, base_width, N, k, dropout, strides):
    x = BatchNormalization(_CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, beta_regularizer=l2(bn_weight_decay),
                           gamma_regularizer=l2(bn_weight_decay))(x)
    x = Activation('relu')(x)

    # ---- first wide resnet basic dropout block ------
    z = Conv2D(base_width * k, (3, 3), strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(x)
    z = Dropout(dropout)(z)
    z = BatchNormalization(_CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, beta_regularizer=l2(bn_weight_decay),
                           gamma_regularizer=l2(bn_weight_decay))(z)
    z = Activation('relu')(z)
    z = Conv2D(base_width * k, (3, 3), strides=(1, 1,), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(z)

    # projection shortcut
    x = Conv2D(base_width * k, (1, 1), strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(x)

    # merge shortcut
    x = Add()([z, x])

    for i in range(N - 1):
        z = BatchNormalization(_CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, beta_regularizer=l2(bn_weight_decay),
                               gamma_regularizer=l2(bn_weight_decay))(x)
        x = Activation('relu')(z)

        z = Conv2D(base_width * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(z)
        z = Dropout(dropout)(z)
        z = BatchNormalization(_CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, beta_regularizer=l2(bn_weight_decay),
                               gamma_regularizer=l2(bn_weight_decay))(z)
        z = Activation('relu')(z)
        z = Conv2D(base_width * k, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(z)

        x = Add()([z, x])

    return x


def build():
    input = Input((32, 32, 3,))
    x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal',
               kernel_regularizer=l2(cnn_weight_decay), use_bias=False)(input)

    x = _build_main_block(x, base_width=16, N=4, k=10, dropout=0.3, strides=(1, 1))
    x = _build_main_block(x, base_width=32, N=4, k=10, dropout=0.3, strides=(2, 2))
    x = _build_main_block(x, base_width=64, N=4, k=10, dropout=0.3, strides=(2, 2))

    x = BatchNormalization(_CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, beta_regularizer=l2(bn_weight_decay),
                           gamma_regularizer=l2(bn_weight_decay))(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8), (1, 1), padding='same')(x)
    x = Flatten()(x)

    output = Dense(100,
                   activation='softmax',
                   kernel_regularizer=l2(fc_weight_decay),
                   bias_regularizer=l2(fc_weight_decay))(x)

    return Model(inputs=input, outputs=output)
