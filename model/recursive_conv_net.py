# Author: Soheil Khorram
# License: Simplified BSD


from keras.layers import Conv1D, Input, MaxPooling1D, Add, UpSampling1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from model.keras_net import KerasNet
from keras import backend as K
from keras.layers import Layer


class ConstMultiplierLayer(Layer):
    def __init__(self, **kwargs):
        super(ConstMultiplierLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


class Net(KerasNet):
    def __init__(self):
        super(Net, self).__init__()

    def parse_arguments(self, parser):
        super(Net, self).parse_arguments(parser)
        # parser.add_argument('-utt-in-dim', default=40, type=int)
        # parser.add_argument('-utt-out-dim', default=3393, type=int)
        parser.add_argument('-emb-layer-num', default=1, type=int)
        parser.add_argument('-recursive-layer-num-in', default=5, type=int)
        parser.add_argument('-recursive-layer-num-out', default=5, type=int)
        parser.add_argument('-kernel-num', default=512, type=int)
        parser.add_argument('-kernel-size', default=5, type=int)
        parser.add_argument('-activation', default='relu', type=str)
        parser.add_argument('-l2', default=0.0, type=float)
        parser.add_argument('-step-size', default=0.0001, type=float)
        parser.add_argument('-epoch-num', default=20, type=int)

    def set_params(self, prm):
        super(Net, self).set_params(prm)
        self.utt_in_dim = prm.utt_in_dim
        self.utt_out_dim = prm.utt_out_dim
        self.emb_layer_num = prm.emb_layer_num
        self.recursive_layer_num_in = prm.recursive_layer_num_in
        self.recursive_layer_num_out = prm.recursive_layer_num_out
        self.step_size = prm.step_size
        self.kernel_num = prm.kernel_num
        self.kernel_size = prm.kernel_size
        self.activation = prm.activation
        self.l2 = prm.l2
        self.epoch_num = prm.epoch_num

    def construct(self):
        inp = Input(shape=(None, self.utt_in_dim))
        all_y = []
        x = inp
        x = self.emb_subnetwork(x)
        for i in range(self.recursive_layer_num_out):
            x = self.recursive_subnetwork(x)
        x = self.out_conv(x)
        outp = x
        self._net = Model(inputs=inp, outputs=outp)
        self._net.compile(optimizer=Adam(lr=self.step_size),
                          loss='categorical_crossentropy')

    def emb_subnetwork(self, x):
        for l in range(self.emb_layer_num):
            x = self.default_conv(x)
        return x

    def recursive_subnetwork(self, x):
        layer = self.default_conv_layer()
        y = x
        outp = y
        for l in range(self.recursive_layer_num_in):
            y = layer(y)
            outp = Add()([outp, ConstMultiplierLayer()(y)])
        return outp

    def default_conv(self, x):
        return self.default_conv_layer()(x)

    def default_conv_layer(self):
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
                      activity_regularizer=None)

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
