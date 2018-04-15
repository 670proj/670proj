import numpy as np
import re
import itertools
from collections import Counter
import pickle



# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def pad_matrix(i):
    file_name = 'data/matrix/' + str(i) + ".txt"

    with open(file_name, 'rb') as f:
        word_list = pickle.load(f)
        if len(word_list)<= 500:
            padding_num = 500 - len(word_list)
            for j in range(padding_num):
                zero_row = [0] * 100
                word_list.append(zero_row)
            return np.expand_dims(np.array(word_list), axis=2)
        else:
            word_list = word_list[0:500]
            return np.expand_dims(np.array(word_list), axis=2)

def pad_matrix0(file_path):
    '''
    Load X from matrix file address
    '''
    with open(file_path, 'rb') as f:
        word_list = pickle.load(f)
        if len(word_list)<= 500:
            padding_num = 500 - len(word_list)
            for j in range(padding_num):
                zero_row = np.zeros(100).tolist()
                word_list.append(zero_row)
            return np.expand_dims(np.array(word_list), axis=2)
        else:
            word_list = word_list[0:500]
            return np.expand_dims(np.array(word_list), axis=2)

def load_X(file_path, triple,nums_file=21457):
    """
        save files into list first
    """
    files = []
    for i in range(nums_file):
        file_name = file_path +'/' + str(i) + ".txt"
        if triple[i]:
            for k in range(3):
                files.append(file_name)
        else:
            files.append(file_name)
            
    return files

def load_Y(file_path,triple):
    import pandas as pd
    df = pd.read_csv(file_path)
    labels = list(df['cat'])
    cats   = list(set(labels))
    res    = []
    for i, label in enumerate(labels):
        index = cats.index(label)
        label = [0] * len(cats)
        label[index] = 1
        if triple[i]:
            for k in range(3):
                res.append(label)
        else:
            res.append(label)
    return res

def triple(file_path):
    y = []
    triple_set = { 'country', '80s',  'classic rock', 'jazz', 'heavy metal', 'folk'}
    import pandas as pd
    df = pd.read_csv(file_path)
    labels = list(df['cat'])
    for i, label in enumerate(labels):
        if label in triple_set:
            y.append(1)
        else:
            y.append(0)
    return y

def one_hot(num, classes):
    label_list = [0]*classes
    label_list[num] = 1
    return label_list


def batch_iter_new(data, batch_size, num_epochs, shuffle=True):
    '''
        Generates a batch iterator according to file indexes
    '''
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # shuffle_indices = np.random.permutation(np.arange(data_size))
            # shuffled_data = data[shuffle_indices]
            shuffled_data  = data
            import random
            random.shuffle(shuffled_data)
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            X_iter  = [pad_matrix0(item[0]) for item in shuffled_data[start_index:end_index]]
            Y_iter  = [item[1] for item in shuffled_data[start_index:end_index]]
            yield list(zip(X_iter, Y_iter))  


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
            print(shuffle_indices)
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
