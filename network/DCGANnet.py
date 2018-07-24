import tensorflow as tf
import numpy as np
from .network import Network


class DCGANnet(Network):
    def __init__(self, net_name, IMG_SHAPE):
        Network.__init__(self, net_name)
        self.net_name = net_name
        self.IMG_HEIGHT = IMG_SHAPE[0]
        self.IMG_WIDTH = IMG_SHAPE[1]
        self.IMG_CHANEL = IMG_SHAPE[2]

    def setup(self, load_net=False):
        if not load_net:
            X = tf.placeholder(tf.float32, shape=[None, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANEL], name='X')
            Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
            on_train = tf.placeholder(tf.bool, name='on_train')
            batch_size = tf.placeholder(tf.int32, name='batch_size')

            with tf.variable_scope(self.net_name):
                G_h0 = tf.matmul(z, weight_var([z.get_shape().as_list()[1], 128 * 5 * 8], name='G_W_line')) + \
                       bias_var([128 * 5 * 8], name='G_b_line')
                G_h0 = tf.reshape(G_h0, shape=[-1, 8, 5, 128])
                G_h0 = normal(G_h0, [128], on_train, 0.5, [0, 1, 2], 'G_sca_line', 'G_off_line', 'G_mea_line',
                              'G_var_line')
                G_h0 = tf.nn.relu(G_h0)

                G_h1 = deconv2d(G_h0, [batch_size, 16, 10, 256], 3, 3, 2, 2, 'G_W_1', 'G_b_1', padding='SAME')
                G_h1 = normal(G_h1, [256], on_train, 0.5, [0, 1, 2], 'G_sca_1', 'G_off_1', 'G_mea_1', 'G_var_1')
                G_h1 = tf.nn.relu(G_h1)

                G_h2 = deconv2d(G_h1, [batch_size, 32, 20, IMG_CHANEL], 3, 3, 2, 2, 'G_W_2', 'G_b_2', padding='SAME')

                G_h2 = tf.nn.tanh(G_h2)
                G_h2 = G_h2 / 2.
                tf.add_to_collection(name='out', value=G_h2)
