import tensorflow as tf
import numpy as np
import gensim
import string
import time
import datetime
import pickle
import csv
import random
import re


class WordProcessing(object):

    def __init__(self, vocabmodel='word2vector3.model', length=500):
        """
            Do some word preprocessing to embeded vector
        """
        self.wordmodel = gensim.models.Word2Vec.load(vocabmodel)
        self.length    = length
        self.regex = re.compile('[%s]' % re.escape(string.punctuation))

    def word_vector(self, lyric):
        '''
            Conver the lyric to word embeding styles with fixed length
        '''
        words = self.word_list_clean(lyric.replace("\n","").split(" "), self.regex)
        matrix = self.to_matrix(words)
        return self.pad_matrix(matrix)

    def word_list_clean(self, word_list,regex):
        clean_list = []
        for word in word_list:
            word = word.lower()
            word = regex.sub('', word)
            clean_list.append(word)
        return(clean_list)

    def to_matrix(self, words_list):
        """
            convert words to vector model
        """
        matrix = []
        for word in words_list:
            if word in self.wordmodel.wv.vocab:
                matrix.append(np.ndarray.tolist(self.wordmodel.wv.__getitem__(word)))
        return matrix

    def pad_matrix(self, word_list):
        """
            pad the matrix to the fixed format
        """
        if len(word_list)<= self.length :
            padding_num =  self.length - len(word_list)
            for j in range(padding_num):
                zero_row = [0] * 100
                word_list.append(zero_row)
            return [np.expand_dims(np.array(word_list), axis=2)]
        else:
            word_list = word_list[0:self.length]
            return [np.expand_dims(np.array(word_list), axis=2)]

class Model(object):

    def __init__(self, modelfile='checkpoints/model-3200', vocabmodel='word2vector3.model'):
        """
            Load the saved cnn model with model checkpoint file modelfile
            Load the vocabulary model with file vocabmodel
        """
        tf.reset_default_graph() 
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(modelfile+'.meta')
        saver.restore(self.sess, modelfile)

        graph = tf.get_default_graph()
        self.input_x   = graph.get_tensor_by_name("input_x:0")
        self.input_y   = graph.get_tensor_by_name("input_y:0")
        self.drop_out  = graph.get_tensor_by_name("dropout_keep_prob:0")
        self.predictions = graph.get_tensor_by_name("output/predictions:0")
        self.accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")


        #load the vocab model
        self.wordmodel = WordProcessing(vocabmodel)

    def prediction(self, lyric):
        cats = ['country', 'folk', 'jazz', 'metal', 'rock']
        feed_dict = {
            self.input_x: self.wordmodel.word_vector(lyric),
            self.drop_out: 1.0
        }
        prediction = self.sess.run(self.predictions, feed_dict)
        return cats[prediction[0]]

    def testAccuracy(self, X, Y):
        feed_dict = {
            self.input_x: X,
            self.input_y: Y,
            self.drop_out: 1.0
        }
        prediction, accuracy = self.sess.run([self.predictions, self.accuracy], feed_dict)
        # print(prediction)
        time_str = datetime.datetime.now().isoformat()
        print("{}: acc {:g}".format(time_str, accuracy))

# triple = data_helpers.triple(filename)
# x = data_helpers.load_X(filenameX,triple)
# y = data_helpers.load_Y(filename,triple)

def TestAccuracy(model, filename, filenameX):
    
    import data_helpers

    x = data_helpers.load_X(filenameX)
    y = data_helpers.load_Y(filename)

    temp_data = list(zip(x, y))
    random.shuffle(temp_data)
    x_shuffled, y_shuffled = zip(*temp_data)

    dev_sample_index = -1 * int(0.06 * float(len(y)))
    #dev_sample_index = -1
    print(dev_sample_index)
    # dev_sample_index = 3
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    x_dev = [data_helpers.pad_matrix0(item) for item in x_dev]

    # del x, y, x_shuffled, y_shuffled
    model.testAccuracy(x_dev, y_dev)


def TestPrediction(model, filename):
    import pandas as pd
    import random

    df = pd.read_csv(filename)
    # index = random.sample(len(df))
    cat = df.iloc[0]['cat']
    lyric = df.iloc[0]['text']
    cat_pred = model.prediction(lyric)
    print("\n\nOrginal cat {}, Predicticted cat {}\n\n".format(cat, cat_pred))

if __name__=='__main__':


    filename = "../songcleaned2.csv"
    filenameX = "../matrix3"
    vocabmodel = '../word2vector3.model'
    modelfile = 'checkpoints/model-3200'
    model = Model(modelfile, vocabmodel)

    TestPrediction(model, filename)
    TestAccuracy(model, filename, filenameX)
    # x = data_helpers.load_X(filenameX)
    # y = data_helpers.load_Y(filename)

    # temp_data = list(zip(x, y))
    # random.shuffle(temp_data)
    # x_shuffled, y_shuffled = zip(*temp_data)

    # dev_sample_index = -1 * int(0.06 * float(len(y)))
    # #dev_sample_index = -1
    # print(dev_sample_index)
    # # dev_sample_index = 3
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    # x_dev = [data_helpers.pad_matrix0(item) for item in x_dev]

    # del x, y, x_shuffled, y_shuffled


    # cat = df.iloc[0]['cat']
    # lyric = df.iloc[0]['text']

    # print(cat)
    # print(lyric)
    # cat = model.prediction(lyric)
    # print(cat)
    # model.testAccuracy(x_dev, y_dev)
# tf.reset_default_graph()  

# sess = tf.Session()
# # Restore variables from disk.
# saver = tf.train.import_meta_graph('runs/1524340804/checkpoints/model-3200.meta')
# # saver.restore(sess, tf.train.latest_checkpoint('cat9/checkpoints'))
# saver.restore(sess,'runs/1524340804/checkpoints/model-3200')

# graph = tf.get_default_graph()


# input_x   = graph.get_tensor_by_name("input_x:0")
# input_y   = graph.get_tensor_by_name("input_y:0")
# drop_out  = graph.get_tensor_by_name("dropout_keep_prob:0")
# # loss     = graph.get_tensor_by_name("loss:0")
# predictions = graph.get_tensor_by_name("output/predictions:0")
# accuracy = graph.get_tensor_by_name("accuracy/accuracy:0")
# # global_step = graph.ge_tensor_by_name("global_step")
# print("Model restored.")
# # name= [n.name for n in tf.get_default_graph().as_graph_def().node]
# # print(name)
# feed_dict = {
#     input_x: x_dev,
#     input_y: y_dev,
#     drop_out: 1.0
# }
# prediction1, accuracy1 = sess.run([predictions, accuracy], feed_dict)
# print(prediction1)
# print(accuracy1)
# # print(prediction)
# time_str = datetime.datetime.now().isoformat()
# print("{}: acc {:g}".format(time_str, accuracy1))

