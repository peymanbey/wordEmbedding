"""
Vector representation of Words
Neural Probablistic Approach
Skip-Gram Model
Github : peymanbey
"""
from __future__ import division

from helper import download, read_data, build_dataset, gen_batch

#import collections
from math import sqrt
import cPickle as pickle
#import os
#import random
#import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow import placeholder, Variable, truncated_normal, reduce_mean

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
vocab_size = 75000

data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size)
# you can delete the 'words' to reduce memory usage
del words
#%%
print 'Most common words: ', count[:5]
print 'Sample data:', '\n', data[:10], '\n', [reverse_dictionary[i] for i in data[:10]]
del i
#%%

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
embed_size = 256  # vector space embedding size
skip_window = 5  # left and right window size
num_skips = 8  # How many times to use an input to generate a label

# Pick random validation set for sampling nearest neighbor .
# By limiting the validation set to lower IDs we use most frequent words 

valid_size = 16  # Random set of words to evaluate similarity
valid_window = 100  # only pick samples in the head of distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative examples to sample

graph = tf.Graph()

with graph.as_default():

    # input data
    train_inputs = placeholder(tf.int32, shape=[batch_size])
    train_labels = placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/gpu:0'):
        # look up embeddings for inputs
        embeddings = Variable(tf.random_uniform([vocab_size,
                                                 embed_size],
                                                 -1.0,
                                                 1.0))
        # build a lookup to find embedding for each id(word index)
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # build NCE loss
        nce_weights = Variable(truncated_normal([vocab_size, embed_size],
                                                stddev=1.0/sqrt(embed_size)))
        nce_biases = Variable(tf.zeros([vocab_size]))

    # compute NCE loss for a batch
    # tf.nce_loss draw negative samples automatically 
    loss = reduce_mean(tf.nn.nce_loss(nce_weights,nce_biases,embed,
                                      train_labels,num_sampled, vocab_size))
    # SGD optimiser
    optimizer1 = tf.train.GradientDescentOptimizer(1).minimize(loss)
    optimizer2 = tf.train.GradientDescentOptimizer(.1).minimize(loss)
    # Compute cosine similarity between minibatch examples and embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings,
                           transpose_b=True)

    # variable initializer
    init = tf.initialize_all_variables()                           
#%%
num_steps = 300000

with tf.Session(graph=graph) as session:
    # initialiaze variables
    init.run()
    print 'initialized'
    
    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_lables, data_index = gen_batch(data, data_index,
                                                           batch_size,
                                                           num_skips,
                                                           skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_lables}
        
        # one step update
        if step <= 150000:
            _, loss_val = session.run([optimizer1, loss], feed_dict=feed_dict)
        else:
            _, loss_val = session.run([optimizer2, loss], feed_dict=feed_dict)
        average_loss += loss_val
        # calculate the average loss over lass n batches
        n = 1000
        if step % n == 0:
            if step > 0 :
                average_loss /= n
            print 'average loss fo last ',n,' batches: ', average_loss
            average_loss = 0
            
        if step == num_steps-1:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest examples
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "nearest to %s are:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s" % (log_str, close_word)
                print log_str
    
    final_embedding = normalized_embeddings.eval()
with open('final_embedding','w') as f:
    pickle.dump((final_embedding,dictionary,reverse_dictionary), f)
#%% 
# Visualisation
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(filename)

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
#low_dim_embs = tsne.fit_transform(final_embedding[:plot_only,:])
low_dim_embs = tsne.fit_transform(final_embedding[np.random.randint(vocab_size, size=100), :])
labels = [reverse_dictionary[i] for i in xrange(plot_only)]
plot_with_labels(low_dim_embs, labels)

