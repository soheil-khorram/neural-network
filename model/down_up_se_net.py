# Author: Soheil Khorram
# License: Simplified BSD


from keras.layers import Conv1D, Input, MaxPooling1D, Add, UpSampling1D, GlobalAveragePooling1D, Reshape, Dense, multiply
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from model.keras_net import KerasNet


class Net(KerasNet):
    def __init__(self):
        super(Net, self).__init__()

    def parse_arguments(self, parser):
        super(Net, self).parse_arguments(parser)
        # parser.add_argument('-utt-in-dim', default=40, type=int)
        parser.add_argument('-se-ratio', default=1, type=int)
        parser.add_argument('-hourglass-num', default=5, type=int)
        parser.add_argument('-layer-num', default=3, type=int)
        parser.add_argument('-kernel-num', default=512, type=int)
        parser.add_argument('-kernel-size', default=5, type=int)
        parser.add_argument('-activation', default='relu', type=str)
        parser.add_argument('-l2', default=0.0, type=float)
        parser.add_argument('-step-size', default=0.0001, type=float)
        parser.add_argument('-epoch-num', default=20, type=int)
        # parser.add_argument('-utt-out-dim', default=3393, type=int)

    def set_params(self, prm):
        super(Net, self).set_params(prm)
        self.utt_in_dim = prm.utt_in_dim
        self.utt_out_dim = prm.utt_out_dim
        self.se_ratio = prm.se_ratio
        self.hourglass_num = prm.hourglass_num
        self.layer_num = prm.layer_num
        self.step_size = prm.step_size
        self.kernel_num = prm.kernel_num
        self.kernel_size = prm.kernel_size
        self.activation = prm.activation
        self.l2 = prm.l2
        self.epoch_num = prm.epoch_num

    def construct(self):
        inp = Input(shape=(None, self.utt_in_dim))
        x = inp
        x = self.default_conv(x)
        x = self.squeeze_excitation(x)
        for hind in range(self.hourglass_num):
            x = self.hourglass(x)
            x = self.squeeze_excitation(x)
        x = self.out_conv(x)
        outp = x
        self._net = Model(inputs=inp, outputs=outp)
        self._net.compile(optimizer=Adam(lr=self.step_size),
                          loss='categorical_crossentropy')

    def hourglass(self, x):
        all_y = []
        for i in range(self.layer_num):
            x = self.default_conv(x)
            x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
            all_y.append(x)
        x = self.default_conv(x)
        for i in range(self.layer_num):
            y = all_y.pop()
            x = Add()([x, y])
            x = UpSampling1D(size=2)(x)
            x = self.default_conv(x)
        return x

    def squeeze_excitation(self, inp):
        filters = inp._keras_shape[-1]
        se = GlobalAveragePooling1D()(inp)
        se = Reshape((1, filters))(se)
        se = Dense(filters // self.se_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
        return multiply([inp, se])

    def default_conv(self, x):
        return Conv1D(filters=self.kernel_num,
                      kernel_size=self.kernel_size,
                      strides=1,
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=1,
                      activation=self.activation,
                      use_bias=True,
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      kernel_regularizer=l2(self.l2),
                      bias_regularizer=l2(self.l2),
                      activity_regularizer=None)(x)

    def out_conv(self, x):
        return Conv1D(filters=self.utt_out_dim,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      data_format='channels_last',
                      dilation_rate=1,
                      activation='softmax',
                      use_bias=True,
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      kernel_regularizer=l2(self.l2),
                      bias_regularizer=l2(self.l2),
                      activity_regularizer=None)(x)
