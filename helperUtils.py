"""Helper functions for word embedding"""
#from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import collections
import math
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

name = download('text8.zip')

#%% 
# store the data into a list
# the list contains the text as sequential words
def read_data(name):
    """Read data from zip file into list of strings"""
    with zipfile.ZipFile(name) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
#%%
words = read_data(name)
print 'Data size', len(words)
#%%
# Build a Dictionary and replace rare words with UNK tokken
# translate the input text in terms of unique numerical IDs
vocab_size = 50000


def build_dataset(words):
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

data, count, dictionary, reverse_dictionary = build_dataset(words)
# you can delete the 'words' to reduce memory usage
del words
#%%
print 'Most common words: ', count[:5]
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])