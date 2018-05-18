import os
import sys
import zipfile
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import re
import seaborn as sns
import pickle
import gzip
import bz2
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def cm_plot(cm, classes=None, title=None, cmap=plt.cm.Reds, figsize=(10,8), normalize=False, cbar=True):
    """reads in a confusion matrix from a scikit_learn CM, returns a matplotlib plot"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    p = plt.figure(figsize=figsize)
    p = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    if title != None:
        p = plt.title(title)
        
    if cbar:
        p = plt.colorbar()
    
    if classes != None:
        tick_marks = np.arange(len(classes))
        p = plt.xticks(tick_marks, classes, rotation=45)
        p = plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        p = plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    p = plt.tight_layout()
    p = plt.ylabel('True label')
    p = plt.xlabel('Predicted label')
    return p

def save_pickle(obj, out_file, zip_class=None):
    if zip_class != None:
        f = zip_class.open(out_file, 'wb')
    else:
        f = open(out_file, 'wb')
    pickle.dump(obj, f)
    f.close()
        
def load_pickle(in_file, zip_class=None):
    if zip_class != None:
        f = zip_class.open(in_file, 'rb')
    else:
        f = open(in_file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret
