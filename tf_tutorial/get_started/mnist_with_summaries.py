from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

# we can't initialize these variable to 0 - the network will get stuck


def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)


def bias_variable(shae):
    inital = tf.constant(0.1, shape=shae)
    return tf.Variable(inital)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)"""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer"""
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('bias'):
            bias = bias_variable([output_dim])
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            preactive = tf.matmul(input_tensor, weights) + bias
            tf.summary.histogram('pre_activation', preactive)
        activations = act(preactive, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # input placeholder
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_input')

    # visualize the input images
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy
        )

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out
    merged = tf.summary.merge_all()

    batch_size = 100
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',
                                             session.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')

        for i in range(FLAGS.max_steps):
            if i % 10 == 0:
                # test
                summary, acc = session.run([merged, accuracy],
                                           feed_dict={
                                               x: mnist.test.images,
                                               y_: mnist.test.labels,
                                               keep_prob: 1
                                           })
                test_writer.add_summary(summary, i)
                print("accuracy at step %s : %s" % (i, acc))
            else:
                if i % 100 == 99:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    xs, ys = mnist.train.next_batch(batch_size)
                    summary, _ = session.run([merged, train_step],
                                             feed_dict={
                                                 x: xs,
                                                 y_: ys,
                                                 keep_prob: FLAGS.dropout
                                             },
                                             options=run_options,
                                             run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print("Adding run metadata for", i)
                else:
                    xs, ys = mnist.train.next_batch(batch_size)
                    summary, _ = session.run([merged, train_step],
                                             feed_dict={
                                                 x: xs,
                                                 y_: ys,
                                                 keep_prob: FLAGS.dropout
                                             })
                    train_writer.add_summary(summary, i)

        train_writer.close()
        test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                             '/tmp/dataset_fl/MNIST_data'),
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default=os.path.join(
                            os.getenv('TEST_TMPDIR', '/tmp'),
                            'tensorflow/mnist/logs/mnist_with_summaries'),
                        help='Summaries log diretory.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)