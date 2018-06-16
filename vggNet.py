import tensorflow as tf
import numpy as np

def conv(input, weights_name, bias_name, parameters, activation="relu"):

    with tf.variable_scope(weights_name[:-2]):
        weights = parameters[weights_name]
        bias = parameters[bias_name]

        conv_no_bias = tf.nn.conv2d(input=input, filter=weights, strides=[0, 2, 2, 0], padding="SAME", name="without_bias")
        conv_z = tf.nn.bias_add(conv_no_bias, bias, name="bias_add")
        conv_a = tf.nn.relu(conv_z, name=weights_name[:-2] + 'relu')

    return conv_a

def dense(input, weights_name,  bias_name, parameters, activation='relu'):

    with tf.variable_scope(weights_name[:-2]):
        weights = parameters[weights_name]
        bias = parameters[bias_name]

        dense_no_bias = tf.layers.dense(inputs=input, units=bias.shape[0], name="without_bias", use_bias=True, kernel_initializer=tf.initializers.zeros)
        dense_kernel = tf.get_default_graph().get_tensor_by_name( 'without_bias' + '/' + 'kernel:0')
        tf.assign(dense_kernel, weights)
        dense_z = tf.nn.bias_add(dense_no_bias, bias, name="bias_add")

        dense_a = tf.nn.relu(dense_z, 'relu')

    return dense_a


def vgg16_model(sess, parameters):

    X = tf.placeholder(dtype=tf.float16, shape=[10, 224, 224, 3], name="input")

    conv1_1 = conv(X, "conv1_1_W", bias_name="conv1_1_b", parameters=parameters)
    conv1_2 = conv(conv1_1, "conv1_2_W", bias_name="conv1_2_b", parameters=parameters)
    conv2_1 = conv(conv1_2, "conv2_1_W", bias_name="conv2_1_b", parameters=parameters)
    conv2_2 = conv(conv2_1, "conv2_2_W", bias_name="conv2_2_b", parameters=parameters)
    conv3_1 = conv(conv2_2, "conv3_1_W", bias_name="conv3_1_b", parameters=parameters)
    conv3_2 = conv(conv3_1, "conv3_2_W", bias_name="conv3_2_b", parameters=parameters)
    conv3_3 = conv(conv3_2, "conv3_3_W", bias_name="conv3_3_b", parameters=parameters)
    conv4_1 = conv(conv3_3, "conv4_1_W", bias_name="conv4_1_b", parameters=parameters)
    conv4_2 = conv(conv4_1, "conv4_2_W", bias_name="conv4_2_b", parameters=parameters)
    conv4_3 = conv(conv4_2, "conv4_3_W", bias_name="conv4_3_b", parameters=parameters)
    conv5_1 = conv(conv4_3, "conv5_1_W", bias_name="conv5_1_b", parameters=parameters)
    conv5_2 = conv(conv5_1, "conv5_2_W", bias_name="conv5_2_b", parameters=parameters)
    conv5_3 = conv(conv5_2, "conv5_3_W", bias_name="conv5_3_b", parameters=parameters)
    conv5_3_flatten = tf.contrib.layers.flatten(conv5_3)
    fc7 = dense(conv5_3_flatten, "fc7_W", bias_name="fc7_b", parameters=parameters)
    fc8 = dense(fc7, "fc8_W", bias_name="fc8_b", parameters=parameters)
    fc9 = tf.nn.softmax(fc8, name="softmax_output")


    init = tf.global_variables_initializer()
    sess.run(init)

    all_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for variable in all_variables:
        print(1)
        print(variable.name)

    for op in tf.get_default_graph().get_operations():
        print(op.name)



def load_weights():
    file = np.load("vgg16_weights.npz")
    keys = file.keys()

    # for key in keys:
    #     print(file[key].shape)
    #     print(key)

    return file

def main(argv):
    parameters = load_weights()
    with tf.Session() as sess:
        vgg16_model(sess, parameters)
if __name__ == '__main__':
    tf.app.run()