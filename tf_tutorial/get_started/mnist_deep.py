"""
   A deep MNIST classifier using convolutional layers.
   reference:
      https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_deep.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf


FLAGS = None


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


def deepnn(x):
    """
    
    :param x: an input tensor with the dimensions (N_examples, 784), 
    where 784 is the number of pixels in a standard MNIST image.
    :return: 
        A tuple (y, keep_prob). y is a tensor of shape (N_samples, 10), 
        with values equal to the logits of classifying the digit into one of 
        10 classes (the digits 0-9). keep-prob is scalar placeholder for the 
        probability of dropout.
    """

    # reshape input to meet the CNN shape
    # the last one column is the color channel.
    # shape - [batch_size, height, width, channel]
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional.
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1.
    # after 2 round of downsampling, our 28x28 image is down to 7x7x64
    # feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation
    # of features.
    with tf.name_scope('droput'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv, keep_prob


def main(_):
    # import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # create model
    x = tf.placeholder(tf.float32, [None, 784])
    y_real = tf.placeholder(tf.float32, [None, 10])
    y_conv, keep_prob = deepnn(x)

    # calculate loss
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_real, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    # set up optimizer
    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
            cross_entropy)

    # calculate accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_real, 1),
                                      tf.argmax(y_conv, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # make saving directory
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    # start session
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # train
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = session.run(accuracy, feed_dict={
                    x: batch[0],
                    y_real: batch[1],
                    keep_prob: 1.0
                })
                print('step %d, training accuracy %g' % (i, train_accuracy))
            _ = session.run(train_step, feed_dict={
                x: batch[0],
                y_real: batch[1],
                keep_prob: 0.5
            })
        # test
        test_accuracy = session.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_real: mnist.test.labels,
            keep_prob: 1.0
        })
        print('test accuracy %g' % test_accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/dataset_fl/MNIST_data',
                        help='Directory for storing input data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
