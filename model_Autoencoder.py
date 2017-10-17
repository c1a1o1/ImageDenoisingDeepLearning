import numpy as np
import tensorflow as tf
from model_helper import *

class Autoencoder(object):
    def __init__(self, learning_rate = 0.0001):
        tf.reset_default_graph()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.train = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name='train')
        self.target = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name='target')
        self.filter_shape = (5,5)

    def build_model(self):

        # self.lowdose = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.train, dtype=tf.float32)
        # self.normaldose = tf.map_fn(lambda img: tf.image.per_image_standardization(img), self.target, dtype=tf.float32)
        ### Encoder
        # layers.conv2d doesn't have input channels as an input like nn.conv2d
        # it automatically set the channels?
        # In convolution, strides larger than 1 downsamples image
        # In deconvolution, strides larger than 2 upsamples image
        # Deconvolution is essentially unsampling convolution
        conv1 = tf.layers.conv2d(self.train, filters=4, kernel_size=self.filter_shape, padding='same', activation=lrelu)
        conv2 = tf.layers.conv2d(conv1, 8, self.filter_shape, strides=1, padding='same', activation=lrelu)
        conv3 = tf.layers.conv2d(conv2, 16, self.filter_shape, strides=1, padding='same', activation=lrelu)
        encoded = tf.layers.conv2d(conv3, 32, self.filter_shape, strides=1, padding='same', activation=lrelu)
        ### Decoder
        deconv1 = tf.layers.conv2d_transpose(encoded, 32, self.filter_shape, strides=1, padding='same', activation=lrelu)
        deconv2 = tf.layers.conv2d_transpose(deconv1, 16, self.filter_shape, strides=1, padding='same', activation=lrelu)
        deconv3 = tf.layers.conv2d_transpose(deconv2, 8, self.filter_shape, strides=1, padding='same', activation=lrelu)
        deconv4 = tf.layers.conv2d_transpose(deconv3, 1, self.filter_shape, padding='same', activation=lrelu)
        logits = lrelu(deconv4 + self.train)
        # logits = deconv4
        # Pass logits through sigmoid to get reconstructed image
        # self.decoded = tf.nn.sigmoid(logits, name="decoded")
        self.decoded = logits
        # add decoded to collection so that it can be located after graph resoration
        tf.add_to_collection("predict", self.decoded)
        # Pass logits through sigmoid and calculate the cross-entropy loss
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)

        ### Using logits to calculate loss is much better than using decoded
        loss = tf.squared_difference(self.decoded, self.target)
        # Get cost and define the optimizer
        self.cost = tf.reduce_sum(loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()