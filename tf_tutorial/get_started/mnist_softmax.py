"""
    A very simple MNIST classifier.
    reference: https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/mnist/mnist_softmax.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    bias = tf.Variable(tf.zeros([10]))
    y_hat = tf.matmul(x, W) + bias

    y_real = tf.placeholder(tf.float32, [None, 10])

    # loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=y_hat))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5)\
        .minimize(cross_entropy)

    with tf.Session() as session:
        tf.global_variables_initializer().run()
        # train
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            session.run(train_step, feed_dict={
                x: batch_xs,
                y_real: batch_ys
            })
        # test
        correct_prediction = tf.equal(tf.argmax(y_real, 1), tf.argmax(y_hat, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(session.run(accuracy, feed_dict={
            x: mnist.test.images,
            y_real: mnist.test.labels
        }))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/dataset_fl/MNIST_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
