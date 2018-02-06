#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import csv

from scipy import misc
import numpy as np

import time
import cv2

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_dir", "data/train", "Data source for the data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/Users/Vivek/Desktop/ml/sbi hackathon/ocr/ocr_mnist/models", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def binarize(img):
    img[img<127] = 0
    img[img>126] = 1
    return img

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def rgb2grey(image):
    if len(image.shape)==2:
        return image
    return cv2.cvtColor( image, cv2.COLOR_RGB2GRAY )

def threshold(img):
    img[img<127] = 0
    img[img>126] = 255
    return img

def preprocess_image(img_path):
    img = misc.imread(img_path)

    img_resized = misc.imresize(img,(28,28))
    img_bw = rgb2grey(img_resized)

    # cv2.imshow('image',img_bw)
    img_bw = threshold(img_bw)
    misc.imsave('test/test.jpg', img_bw)

    img_binary = binarize(img_bw)

    return img_binary

def load_data_and_labels(data_dir):

    x_img = []
    x_imgs_path = []
    y = []

    for root,dirs,files in os.walk(data_dir):

        noOfLabels = len(dirs)
        curLabel = -1
        print(dirs)
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
                print(img.shape)
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


test_img_path = 'test/a_2.png'
y_test = []

x_img = [preprocess_image(test_img_path)]
x_test = np.array(x_img, dtype=np.float32)

# cv2.imshow(x_img[0])
print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout/keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("fc2/yconv").outputs[0]


        start_time = time.time()
        batch_scores = sess.run(scores, {input_x: x_img, dropout_keep_prob: 1.0})
        time_taken = time.time()-start_time

        print(batch_scores)
        print(time_taken)