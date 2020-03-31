import os
import re
import numpy as np

''' 
  Utilities for saving and loading features and labels numpy arrays
  Assumes extensions are .npy
  Suffix for features and labels can be specfied e.g <f_name>_label.data
'''

def save_array(arr_in, path, suffix=''):
    ''' Saves numpy array to path 
        a: the numpy array to save
        path: the path to the file
        suffix: suffix to append to the end of the filename
    '''
    assert os.path.isdir(os.path.dirname(path)), "path to array does not exist"
    np.save(path.replace('.npy', suffix + '.npy'), arr_in)

def load_array(path, suffix=''):
    ''' Load .npy file
    '''
    assert os.path.isdir(os.path.dirname(path)), "path to array does nnot exists" 
    arr_in = np.load(path.replace('.npy', suffix + '.npy'))
    return arr_in
