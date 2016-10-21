"""
Helper Functions
"""
from __future__ import division
import collections
from math import sqrt
import os
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#%%
# download the data
url = 'http://mattmahoney.net/dc/'


def download(name):
    """download the file if not present"""
    if not os.path.exists(name):
        name, _ = urllib.request.urlretrieve(url+name, name)
    return name


def read_data(name):
    """Read data from zip file into list of strings"""
    with zipfile.ZipFile(name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
    

def build_dataset(words, vocab_size):
    """
    """
    # add UNK to the list of words
    count = [('UNK', -1)]
    # extract vocab_size most common words and add them to list of words
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    # build a dictionary of words extracted and assign a unique ID to each
    dictionary = dict()
    for word, _ in count:
        # to assign a unique ID we use the length of the dictionary,  as it is
        # growing each time a new entry is added and that the words in the
        # count list are unique, using len(dictionary) as the ID is sound
        dictionary[word] = len(dictionary)
    # initiate the data list
    data = list()
    unk_count = 0
    # loop through all the text, replace each word by the unique ID assigned
    # to it. Note that only vocab_size-1 most frequent words habe unique IDs.
    # The rest will be replaced by ID(UNK)
    for word in words:
        # if the word exists in the vocabulary replace it by it's unique index
        if word in dictionary:
            index = dictionary[word]
        # for any other word/token replace it with 0
        else:
            index = 0
            unk_count += 1
        # add the ID to the transformed dataset 'data'
        data.append(index)
    # store the number of appearance of 'UNK'
    count[0] = ('UNK', unk_count)
    # reverse dictionary
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data_index = 0

# generate trainign batches for SkipGram model
def gen_batch(data, data_index,
              batch_size, num_skips, skip_window):
    """"""

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # initialize batch and labels
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span is the length of skip_window+target+skip_window
    span = 2 * skip_window + 1
    # init a buffer with length span
    buf = collections.deque(maxlen=span)

    # read the first bit of the text, length = span, into the buffer
    for _ in xrange(span):
        buf.append(data[data_index])
        # use indexing with %len(data) for doing the indexing correctly after
        # on loop of the text. This is needed for more than one epoch of
        # training
        data_index = (data_index + 1) % len(data)

    # chech this
    for i in xrange(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_avoid = [skip_window]
        for j in xrange(num_skips):
            while target in targets_avoid:
                target = random.randint(0, span-1)
            targets_avoid.append(target)
            batch[i * num_skips + j] = buf[skip_window]
            labels[i * num_skips + j, 0] = buf[target]
        buf.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels, data_index