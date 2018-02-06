from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import os
from scipy import misc
import random
import numpy as np
import json

FLAGS = None
checkpoint_every = 100
checkpoint_prefix = 'models_large_2/model'
eval_every = 500
num_epochs = 100
train_dir = 'processed_generated_imgs_mnist_2/train'
test_dir = 'processed_generated_imgs_mnist_2/test'


def binarize(img):
    img[img<127] = 0
    img[img>126] = 1
    return img

def load_data_and_labels(data_dir):

    x_img = []
    x_imgs_path = []
    y = []

    for root,dirs,files in os.walk(data_dir):

        noOfLabels = len(dirs)
        curLabel = -1
        print(dirs)
        with open('labels_order.json', 'w') as f:
          json.dump(dirs, f)

        for _dir in dirs:
            curLabel += 1

            dir_path = os.path.join(root,_dir)
            print(dir_path)
            for file in os.listdir(dir_path):
                if '.DS_Store' in file:
                    continue

                img_path = os.path.join(dir_path,file)
                img = misc.imread(img_path)
                # img = np.divide(img,255)
                img = binarize(img)
                # print(img.shape)
                x_img.append(img)
                x_imgs_path.append(img_path)

                arr = []
                for _ in range(noOfLabels):
                    arr.append(0)
                arr[curLabel] = 1

                if len(y) == 0:
                    y = [arr]
                else:
                    y = np.concatenate([y,[arr]],0) 
                  

    return [x_img, y, x_imgs_path]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

x_img, y, _ = load_data_and_labels(train_dir)
# print(x_img)
x_train = np.array(x_img,dtype=np.float32)
y_train = np.array(y,dtype=np.float32)

x_img, y_test, x_img_paths = load_data_and_labels(test_dir)
y_test = np.argmax(y_test, axis=1)
x_test = np.array(x_img, dtype=np.float32)

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.expand_dims(x, -1)
    # x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 62])
    b_fc2 = bias_variable([62])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2), b_fc2, name="yconv")
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Create the model
x = tf.placeholder(tf.float32, [None, 28, 28], name="input_x")

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 62])

# Build the graph for the deep net
y_conv, keep_prob = deepnn(x)

with tf.name_scope('loss'):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                          logits=y_conv)
cross_entropy = tf.reduce_mean(cross_entropy)

with tf.name_scope('adam_optimizer'):
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

graph_location = tempfile.mkdtemp()
print('Saving graph to: %s' % graph_location)
train_writer = tf.summary.FileWriter(graph_location)
train_writer.add_graph(tf.get_default_graph())

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

  batches = batch_iter(
      list(zip(x_train, y_train)), 64, num_epochs)
  # Training loop. For each batch...
  count = 0
  for batch in batches:
    count += 1
    x_batch, y_batch = zip(*batch) 
    if count % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: x_batch, y_: y_batch, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (count, train_accuracy))
    train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5}) 

    if count % checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=count)
        print("Saved model checkpoint to {}\n".format(path))

    if count % eval_every == 0:
      batches_test = batch_iter(
          list(zip(x_test, y_test)), 64, 1)
      accuracy_arr = []
      for batch_test in batches_test:
        x_test_batch, y_test_batch = zip(*batch) 
        accuracy_arr.append(accuracy.eval(feed_dict={
          x: x_test_batch, y_: y_test_batch, keep_prob: 1.0}))

      acc_sum = 0
      for acc in accuracy_arr:
        acc_sum += acc
      accuracy_avg = acc_sum/len(accuracy_arr) 
      print("Test accuracy")
      print(accuracy_avg)

