#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
import pickle
import csv
import random

res_file="result.txt"
tra_file="train.txt"

# Parameters
# ==================================================

def train_step(x_batch, y_batch):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: dropout_keep_prob
    }
    _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    res = str(time_str) + ", " + str(step) + ", " + str(loss) + ", " + str(accuracy) + '\n'
    file2.write(res)
    train_summary_writer.add_summary(summaries, step)

def dev_step(x_batch, y_batch, writer=None):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }
    step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
    res = str(time_str) + ", " + str(step) + ", " + str(loss) + ", " + str(accuracy) + '\n'
    file.write(res)
    if writer:
        writer.add_summary(summaries, step)


# Data Preparation
# ==================================================

# Load data

print("Loading data...")
#x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
"""file_name = "0.txt"
x = []
with open(file_name, 'rb') as f:
    x.append(pickle.load(f))"""
spli_percentage = 0.01
num_checkpoints = 100
batch_size      = 64
num_epochs      = 10
evaluate_every  = 100
checkpoint_every= 100
dropout_keep_prob = 0.5
filename = "../app/songcleaned2.csv"
filenameX = "/Users/HJK-BD/Downloads/model/matrix3"
x = data_helpers.load_X(filenameX)
y = data_helpers.load_Y(filename)
# x=[]
# x.append(data_helpers.pad_matrix(0))
# x.append(data_helpers.pad_matrix(1))
# x.append(data_helpers.pad_matrix(2))
# x.append(data_helpers.pad_matrix(3))
# x.append(data_helpers.pad_matrix(4))
# x.append(data_helpers.pad_matrix(5))
# x.append(data_helpers.pad_matrix(6))
# triple = data_helpers.triple(filename)
# x = data_helpers.load_X(filenameX,triple)
# y = data_helpers.load_Y(filename,triple)
# category = {'rock': 0, 'pop': 1, 'classic rock': 2, 
#             'country': 3, 'hard rock': 4, 'jazz': 5,
#             'folk': 6, '80s': 7, 'heavy metal': 8}
# count = 0
# with open(filename) as f:
#     f_csv = csv.reader(f)
#     headers = next(f_csv)
#     for row in f_csv:
#         if count <=6:
#             y.append(data_helpers.one_hot(category[row[3]], len(category)))
#             count += 1
#         else:
#             break


# Build vocabulary
#max_document_length = max([len(x.split(" ")) for x in x_text])
#vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
#x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
temp_data = list(zip(x, y))
random.shuffle(temp_data)
x_shuffled, y_shuffled = zip(*temp_data)
# np.random.seed(10)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
#x_shuffled = x[shuffle_indices]
#y_shuffled = y[shuffle_indices]
# x_shuffled = x
# y_shuffled = y
# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(spli_percentage * float(len(y)))
print(dev_sample_index)
# dev_sample_index = 3
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

x_dev = [data_helpers.pad_matrix0(item) for item in x_dev]

del x, y, x_shuffled, y_shuffled

#print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

        # Write vocabulary
        #vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        file = open(res_file, 'w') 
        file2 = open(tra_file, 'w')

        # Generate batches
        batches = data_helpers.batch_iter_new(
            list(zip(x_train, y_train)), batch_size, num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

        file.close()
        file2.close()
