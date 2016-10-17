"""Helper functions for word embedding"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
# read the data into a list
 def read_data(name):
     """Extract the file enclosed in a zip as list of words"""
     with zipfile.Zipfile(name) as f:
         data = tf.