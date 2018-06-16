import tensorflow as tf
import numpy as np
import os
import datetime

def conv(input, weights_name, bias_name, parameters, activation="relu", pool=False):

    with tf.variable_scope(weights_name[:-2]):
        weights = parameters[weights_name]
        bias = parameters[bias_name]

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[0, 1, 1, 0], padding="SAME", name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name='relu')

        if pool:
            print("pool")
            print(conv_a.get_shape().as_list())
            conv_pool = tf.layers.max_pooling2d(conv_a, pool_size=[2,2], strides=2)
            return conv_pool


    return conv_a

def dense(input, weights_name,  bias_name, parameters, activation='relu'):

    with tf.variable_scope(weights_name[:-2]):
        weights = parameters[weights_name]
        bias = parameters[bias_name]

        dense_no_bias = tf.layers.dense(inputs=input, units=bias.shape[0], name="without_bias", use_bias=True, kernel_initializer=tf.initializers.zeros)
        dense_kernel = tf.get_default_graph().get_tensor_by_name( weights_name[:-2] + '/without_bias' + '/' + 'kernel:0')
        tf.assign(dense_kernel, weights)
        dense_z = tf.nn.bias_add(dense_no_bias, bias, name="bias_add")

        dense_a = tf.nn.relu(dense_z, 'relu')

    return dense_a


def vgg16_model(sess, parameters):

    root_logdir = "tf_logs"
    log_dir = os.path.join(root_logdir, str(datetime.datetime.now()))

    X = tf.placeholder(dtype=tf.float16, shape=[None, 224, 224, 3], name="input")

    conv1_1 = conv(X, "conv1_1_W", bias_name="conv1_1_b", parameters=parameters)
    conv1_2 = conv(conv1_1, "conv1_2_W", bias_name="conv1_2_b", parameters=parameters, pool=True)

    conv2_1 = conv(conv1_2, "conv2_1_W", bias_name="conv2_1_b", parameters=parameters)
    conv2_2 = conv(conv2_1, "conv2_2_W", bias_name="conv2_2_b", parameters=parameters, pool=True)
    conv3_1 = conv(conv2_2, "conv3_1_W", bias_name="conv3_1_b", parameters=parameters)
    conv3_2 = conv(conv3_1, "conv3_2_W", bias_name="conv3_2_b", parameters=parameters)
    conv3_3 = conv(conv3_2, "conv3_3_W", bias_name="conv3_3_b", parameters=parameters, pool=True)
    conv4_1 = conv(conv3_3, "conv4_1_W", bias_name="conv4_1_b", parameters=parameters)
    conv4_2 = conv(conv4_1, "conv4_2_W", bias_name="conv4_2_b", parameters=parameters)
    conv4_3 = conv(conv4_2, "conv4_3_W", bias_name="conv4_3_b", parameters=parameters, pool=True)
    conv5_1 = conv(conv4_3, "conv5_1_W", bias_name="conv5_1_b", parameters=parameters)
    conv5_2 = conv(conv5_1, "conv5_2_W", bias_name="conv5_2_b", parameters=parameters)
    conv5_3 = conv(conv5_2, "conv5_3_W", bias_name="conv5_3_b", parameters=parameters, pool=True)
    conv5_3_flatten = tf.contrib.layers.flatten(conv5_3)
    fc6 = dense(conv5_3_flatten, "fc6_W", bias_name="fc7_b", parameters=parameters)
    fc7 = dense(fc6, "fc7_W", bias_name="fc7_b", parameters=parameters)
    fc8 = dense(fc7, "fc8_W", bias_name="fc8_b", parameters=parameters)
    fc9 = tf.nn.softmax(fc8, name="softmax_output")

    Y_hat = tf.argmax(fc9, axis=0, name="Y_hat")

    Y_hat_summary = tf.summary.tensor_summary(name="Y_hat_summary", tensor=Y_hat)
    file_write = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    init = tf.global_variables_initializer()
    sess.run(init)

    # saver = tf.train.Saver()
    # saver.save(sess, os.path.join(os.getcwd(), "model"))



def load_weights():
    file = np.load("vgg16_weights.npz")
    keys = file.keys()

    for key in keys:
        print(file[key].shape)
        print(key)

    print(len(keys))
    return file

def main(argv):


    parameters = load_weights()
    with tf.Session() as sess:
        vgg16_model(sess, parameters)


if __name__ == '__main__':
    tf.app.run()