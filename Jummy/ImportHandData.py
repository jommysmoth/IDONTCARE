import numpy as np
import scipy
import scipy.io as sio
import itertools
'''
Author: Lance Wrobel
'''

def load_hand_signals():
    '''
    Import each subjects' dictionaries into one list
    Remove any metadata in the dictionaries
    '''
    subject_signals = []
    
    signal_1 = sio.loadmat("female_1.mat")
    n = len(signal_1.keys())
    subject_signals.append(dict(itertools.islice(signal_1.items(),3,n))) #the three removes the hand metadata
    signal_2 = sio.loadmat("female_2.mat")
    subject_signals.append(dict(itertools.islice(signal_2.items(),3,n))) 
    signal_3 = sio.loadmat("female_3.mat")
    subject_signals.append(dict(itertools.islice(signal_3.items(),3,n)))
    signal_4 = sio.loadmat("male_1.mat")
    subject_signals.append(dict(itertools.islice(signal_4.items(),3,n)))
    signal_5 = sio.loadmat("male_2.mat")
    subject_signals.append(dict(itertools.islice(signal_5.items(),3,n)))
    
    return subject_signals

def get_signal_classes(subject_signals):

    n = len(subject_signals)
    all_signals_classes = []
    
    for i in range(n):

        current_subject = subject_signals[i]
        class_keys = list(current_subject.keys())
        repeated_list = list(itertools.chain.from_iterable(itertools.repeat(x, 30) for x in class_keys)) #30 trials, generalize
        all_signals_classes.extend(repeated_list) 
        
    return all_signals_classes

def get_combined_channel_classes():

    y = []
    class_keys_new = ['cyl','hook','tip','palm','spher','lat']
    for i in range(5):
        repeated_list = list(itertools.chain.from_iterable(itertools.repeat(x, 30) for x in class_keys_new))
        y.extend(repeated_list)
    return y
