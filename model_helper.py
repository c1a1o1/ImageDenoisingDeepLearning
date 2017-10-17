import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs import vgg16
import numpy as np

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# def lrelu(x, alpha=0.5):
#     return tf.maximum(x, alpha * x)

def conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name, activation_function=lrelu, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        # Filter has four dimension: [filter_height, filter_width, in_channels, out_channels]
        # channels are like thickness of the image
        # (filter_height, filter_width, in_channels) defines size and thickness the filter: (x, y, thickness)
        # The whole 3D filter dot product with 3D patch of input image, and produces only one value
        # The out_channels defines number of filters, or and determines thickness/channels of output
        # (only if the channels/thickness of both input and filter are the same)
        # The output slice size depends on stride, when stride = 1, size doesn't change
        # The input has four dimension: [batch, in_height, in_width, in_channels]
        # The first dimension defines batch size, the filtering is performed on each "batch" independently
        # and the output has the same batch size

        filter = tf.Variable(tf.random_normal([ksize, ksize, in_channels, out_channels], stddev=0.03))
        output = tf.nn.conv2d(input_image, filter, strides=[1, stride, stride, 1], padding='SAME')
        output = slim.batch_norm(output)
        if activation_function:
            output = activation_function(output)
        return output, filter

def residual_layer(input_image, ksize, in_channels, out_channels, stride, scope_name):
    with tf.variable_scope(scope_name):
        output, filter = conv_layer(input_image, ksize, in_channels, out_channels, stride, scope_name+"_conv1")
        output, filter = conv_layer(output, ksize, out_channels, out_channels, stride, scope_name+"_conv2")
        output = tf.add(output, tf.identity(input_image))
        return output, filter

def transpose_deconvolution_layer(input_tensor, used_weights, new_shape, stride, scope_name):
    with tf.varaible_scope(scope_name):
        output = tf.nn.conv2d_transpose(input_tensor, used_weights, output_shape=new_shape, strides=[1, stride, stride, 1], padding='SAME')
        output = tf.nn.relu(output)
        return output

def resize_deconvolution_layer(input_tensor, new_shape, scope_name):
    with tf.variable_scope(scope_name):
        output = tf.image.resize_images(input_tensor, (new_shape[1], new_shape[2]), method=1)
        output, unused_weights = conv_layer(output, 3, new_shape[3]*2, new_shape[3], 1, scope_name+"_deconv")
        return output

def deconvolution_layer(input_tensor, new_shape, scope_name):
    return resize_deconvolution_layer(input_tensor, new_shape, scope_name)

def output_between_zero_and_one(output):
    output +=1
    return output/2

def get_pixel_loss(target,prediction):
    pixel_difference = target - prediction
    pixel_loss = tf.nn.l2_loss(pixel_difference)
    return pixel_loss

def get_style_layer_vgg16(image):
    net = vgg16.get_vgg_model()
    style_layer = 'conv2_2/conv2_2:0'
    feature_transformed_image = tf.import_graph_def(
        net['graph_def'],
        name='vgg',
        input_map={'images:0': image},return_elements=[style_layer])
    feature_transformed_image = (feature_transformed_image[0])
    return feature_transformed_image

def get_style_loss(target, prediction):
    target = tf.image.grayscale_to_rgb(target)
    prediction = tf.image.grayscale_to_rgb(prediction)
    feature_transformed_target = get_style_layer_vgg16(target)
    feature_transformed_prediction = get_style_layer_vgg16(prediction)
    feature_count = tf.shape(feature_transformed_target)[3]
    style_loss = tf.reduce_sum(tf.square(feature_transformed_target-feature_transformed_prediction))
    style_loss = style_loss/tf.cast(feature_count, tf.float32)
    return style_loss

def get_smooth_loss(image):
    batch_count = tf.shape(image)[0]
    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]
    channels = 1

    horizontal_normal = tf.slice(image, [0, 0, 0, 0], [batch_count, image_height, image_width-1, channels])
    horizontal_one_right = tf.slice(image, [0, 0, 1, 0], [batch_count, image_height, image_width-1, channels])
    vertical_normal = tf.slice(image, [0, 0, 0, 0], [batch_count, image_height-1, image_width, channels])
    vertical_one_right = tf.slice(image, [0, 1, 0, 0], [batch_count, image_height-1, image_width, channels])
    smooth_loss = tf.nn.l2_loss(horizontal_normal-horizontal_one_right)+tf.nn.l2_loss(vertical_normal-vertical_one_right)
    return smooth_loss