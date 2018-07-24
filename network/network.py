import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average


def layer(op):
    def layer_decorate(self, *args, **kwargs):
        self.input = self.pre_process_tensor
        self.pre_process_tensor = op(self, *args, **kwargs)
        return self

    return layer_decorate


class Network(object):
    def __init__(self, net_name):
        self.net_name = net_name

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def feed(self, input):
        self.input = input

    def normal(self, x, shape, on_train, decay, axes, name_scale, name_offset, name_mean, name_var):
        # batch-normalization
        scale = tf.get_variable(name_scale, shape, initializer=tf.ones_initializer(), trainable=True)
        offset = tf.get_variable(name_offset, shape, initializer=tf.zeros_initializer(), trainable=True)
        variance_epsilon = 1e-7
        mean_p = tf.get_variable(name_mean, shape, initializer=tf.zeros_initializer(), trainable=False)
        var_p = tf.get_variable(name_var, shape, initializer=tf.ones_initializer(), trainable=False)

        # moving average
        def mean_var_with_update():
            mean_ba, var_ba = tf.nn.moments(x, axes, name='moments')
            with tf.control_dependencies([assign_moving_average(mean_p, mean_ba, decay),
                                          assign_moving_average(var_p, var_ba, decay)]):
                return tf.identity(mean_ba), tf.identity(var_ba)

        # with tf.variable_scope('EMA'):
        mean, var = tf.cond(on_train, mean_var_with_update, lambda: (mean_p, var_p))

        return tf.nn.batch_normalization(x, mean, var, offset, scale, variance_epsilon)

    def weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0., stddev=0.02))

    def bias_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

    @layer
    def deconv2d(self, input, output_size, k_h, k_w, d_h, d_w, name_W, name_b, padding='SAME'):
        w = self.weight_var([k_h, k_w, output_size[-1], input.get_shape().as_list()[-1]], name=name_W)
        deconv = tf.nn.conv2d_transpose(input, w, output_shape=output_size, strides=[1, d_h, d_w, 1], padding=padding) + \
                 self.bias_var([output_size[-1]], name=name_b)
        return deconv

    @layer
    def conv2d(self, input, output_dim, k_h, k_w, d_h, d_w, name_W, name_b, padding='SAME'):
        w = self.weight_var([k_h, k_w, input.get_shape().as_list()[-1], output_dim], name=name_W)
        conv = tf.nn.conv2d(input, w, strides=[1, d_h, d_w, 1], padding=padding) + \
               self.bias_var([output_dim], name=name_b)
        return conv

    @layer
    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    @layer
    def mulfc(self, input, output_dim, name_W, name_b, padding='SAME'):
        w = self.weight_var([input.get_shape().as_list()[-1], output_dim], name=name_W)
        fc = tf.matmul(input, w) + self.bias_var([output_dim], name=name_b)
        return fc
