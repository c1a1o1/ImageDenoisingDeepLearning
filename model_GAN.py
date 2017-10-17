from model_helper import *

class GAN(object):
    def __init__(self, learning_rate = 0.0001):
        tf.reset_default_graph()
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.learning_rate = learning_rate
        self.train = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name='train')
        self.target = tf.placeholder(shape=[None, None, None, 1], dtype=tf.float32, name='target')
        self.filter_shape = (5, 5)

    def build_model(self):
        self.Gz = self.generator(self.train)
        self.Dx = self.discriminator(self.target)
        self.Dg = self.discriminator(self.Gz, reuse=True)
        tf.add_to_collection("predict", self.Gz)

        # self.d_loss = -tf.reduce_mean(tf.log(self.Dx) + tf.log(1. - self.Dg))
        self.d_loss = tf.reduce_sum(tf.squared_difference(self.Dx, self.Dg))
        # self.g_loss = tf.metrics.mean_absolute_error(self.target, self.Gz)
        self.g_loss =  tf.reduce_sum(tf.squared_difference(self.target, self.Gz))/20000 + tf.reduce_sum(np.abs(self.target-self.Gz)) / 130000 \
                    -tf.reduce_mean(tf.log(self.Dg)) / 3 \
                     # + STYLE_LOSS_FACTOR * get_style_loss(self.train, self.Gz) #+ SMOOTH_LOSS_FACTOR * get_smooth_loss(self.Gz)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        # Global_step adds 1 every time the graph sees a batch
        self.d_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=d_vars,
                                                                            global_step=self.global_step)
        self.g_solver = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars)
        self.init = tf.global_variables_initializer()

    def generator(self, input):
        # conv1, conv1_weights = conv_layer(input, 3, 1, 8, 1, "g_conv1")
        # conv2, conv2_weights = conv_layer(conv1, 3, 8, 16, 1, "g_conv2")
        # conv3, conv3_weights = conv_layer(conv2, 3, 16, 16, 1, "g_conv3")
        #
        # res1, res1_weights = residual_layer(conv3, 3, 16, 16, 1, "g_res1")
        # res2, res2_weights = residual_layer(res1, 3, 16, 16, 1, "g_res2")
        # res3, res3_weights = residual_layer(res2, 3, 16, 16, 1, "g_res3")
        #
        # # in_channels = 32 / 2
        # # out_channels = in_channels
        # # deconv1 = deconvolution_layer(conv3, [BATCH_SIZE, 28, 28, 16], 'g_deconv1')
        # # deconv2 = deconvolution_layer(deconv1, [BATCH_SIZE, 28, 28, 8], "g_deconv2")
        # #
        # # deconv2 = deconv2 + conv1
        #
        # conv4, conv4_weights = conv_layer(res3, 3, 16, 1, 1, "g_conv4", activation_function=tf.nn.tanh)
        # conv5 = conv4 + input
        # output = tf.nn.sigmoid(conv5)

        conv1 = tf.layers.conv2d(input, filters=4, kernel_size=self.filter_shape, padding='same', activation=lrelu, name='g_conv1')
        conv2 = tf.layers.conv2d(conv1, 8, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_conv2')
        conv3 = tf.layers.conv2d(conv2, 16, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_conv3')
        encoded = tf.layers.conv2d(conv3, 32, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_encoded')
        ### Decoder
        deconv1 = tf.layers.conv2d_transpose(encoded, 32, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_deconv1')
        deconv2 = tf.layers.conv2d_transpose(deconv1, 16, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_deconv2')
        deconv3 = tf.layers.conv2d_transpose(deconv2, 8, self.filter_shape, strides=1, padding='same', activation=lrelu, name='g_deconv3')
        deconv4 = tf.layers.conv2d_transpose(deconv3, 1, self.filter_shape, padding='same', activation=lrelu, name='g_deconv4')
        # output = lrelu(deconv4 + self.train, name='g_output')
        # Pass logits through sigmoid to get reconstructed image
        # self.decoded = tf.nn.sigmoid(logits, name="decoded")
        return deconv4

    def discriminator(self, input, reuse=False):
        conv1, conv1_weights = conv_layer(input, 5, 1, 4, 2, "d_conv1", reuse=reuse)
        conv2, conv2_weights = conv_layer(conv1, 5, 4, 8, 2, "d_conv2", reuse=reuse)
        conv3, conv3_weights = conv_layer(conv2, 5, 8, 16, 2, "d_conv3", reuse=reuse)
        conv4, conv4_weights = conv_layer(conv3, 5, 16, 16, 1, "d_conv4", reuse=reuse)
        conv5, conv5_weights = conv_layer(conv4, 5, 16, 1, 1, "d_conv5", activation_function=tf.nn.sigmoid, reuse=reuse)
        return conv5
