"""
Vector representation of Words
Neural Probablistic Approach
Skip-Gram Model
Theano Implementation
Github : peymanbey
"""
from __future__ import division
from helper import download, read_data, build_dataset, gen_batch
from math import sqrt
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#%%
name = download('text8.zip')
#%% 
# store the data into a list
# the list contains the text as sequential words
words = read_data(name)
print 'Data size', len(words)
#%%
# Build a Dictionary and replace rare words with UNK tokken
# translate the input text in terms of unique numerical IDs
vocab_size = 50000
data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size)
#%%
# you can delete the 'words' to reduce memory usage
del words
print 'Most common words: ', count[:5]
print 'Sample data:', '\n', data[:10], '\n', [reverse_dictionary[i] for i in data[:10]]
del i
#%%
data_index = 0
#%%
batch, labels, data_index = gen_batch(data, data_index,
                                      batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])
#%%
# Build and train skipgram

batch_size = 128  # training batch size
embed_size = 128  # vector space embedding size
skip_window = 1  # left and right window size
num_skips = 2  # How many times to use an input to generate a label

# Pick random validation set for sampling nearest neighbor .
# By limiting the validation set to lower IDs we use most frequent words 

valid_size = 16  # Random set of words to evaluate similarity
valid_window = 100  # only pick samples in the head of distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample
