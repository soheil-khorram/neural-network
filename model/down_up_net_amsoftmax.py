from keras.layers import Conv1D, Input, MaxPooling1D, Add, UpSampling1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from model.keras_net import KerasNet
from keras import backend as K
from keras.layers import Layer


class NormalizedDense(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NormalizedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.kernel = K.l2_normalize(self.kernel, 0)
        super(NormalizedDense, self).build(input_shape)

    def call(self, x):
        x = K.l2_normalize(x, -1)
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)


def amsoftmax_loss(y_ture, y_pred, scale=20, margin=0.1):
    y_pred = y_ture * (y_pred - margin) + (1 - y_ture) * y_pred
    y_pred *= scale
    return K.categorical_crossentropy(y_ture, y_pred, from_logits=True)


class Net(KerasNet):
    def __init__(self):
        super(Net, self).__init__()

    def parse_arguments(self, parser):
        super(Net, self).parse_arguments(parser)
        # parser.add_argument('-utt-in-dim', default=40, type=int)
        parser.add_argument('-hourglass-num', default=5, type=int)
        parser.add_argument('-layer-num', default=5, type=int)
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
        self.hourglass_num = prm.hourglass_num
        self.layer_num = prm.layer_num
        self.step_size = prm.step_size
        self.kernel_num = prm.kernel_num
        self.kernel_size = prm.kernel_size
        self.activation = prm.activation
        self.l2 = prm.l2
        self.utt_out_dim = prm.utt_out_dim
        self.epoch_num = prm.epoch_num

    def construct(self):
        inp = Input(shape=(None, self.utt_in_dim))
        all_y = []
        x = inp
        x = self.default_conv(x)
        for hind in range(self.hourglass_num):
            for i in range(self.layer_num):
                x = self.default_conv(x)
                x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
                all_y.append(x)
            x = self.default_conv(x)
            for i in range(self.layer_num):
                x = Add()([x, all_y.pop()])
                x = UpSampling1D(size=2)(x)
                x = self.default_conv(x)
        # x = self.out_conv(x)
        x = NormalizedDense(self.utt_out_dim)(x)
        outp = x
        self._net = Model(inputs=inp, outputs=outp)
        self._net.compile(optimizer=Adam(lr=self.step_size),
                          loss=amsoftmax_loss)

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
