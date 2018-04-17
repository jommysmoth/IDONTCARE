"""
Loading in Data.

Raw data for now, to be used in RNN
"""
from __future__ import division
import ImportHandData
import numpy as np
from statsmodels.tsa.ar_model import AR
from scipy.signal import decimate
import matplotlib.pyplot as plt


def makenp(signal_per, label, chn_amount):
    """HEHE."""
    label_array = np.empty([len(chn_amount),
                           signal_per[label + '_ch1'].shape[0],
                           signal_per[label + '_ch1'].shape[1]])
    ch_label = [label + '_ch' + str(x) for x in chn_amount]
    for li, x in enumerate(ch_label):
        label_array[li, :, :] = signal_per[x]
    return label_array


def makefullset(signal, labels, ch_list):
    """Hah."""
    sets = []
    for label in labels:
        label_signal = [makenp(signal[people], label, ch_list) for people in range(5)]
        full_training_set_label = np.stack(label_signal)  # Person - Channels - Samples - Data Points
        sets.append(full_training_set_label)
    complete_set = np.stack(np.asarray(sets))   # Label - Person - Channels - Samples - Data Points
    return complete_set


def ar_fit(sig):
    """Trying Dim Reduction."""
    ar_mod = AR(sig)
    ar_res = ar_mod.fit(trend='nc')
    ar_coefficients = ar_res.params
    return ar_coefficients


def movingaverage(values, window):
    """MA func."""
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def alt_training(training_set):
    """HHEE."""
    alt_index = [training_set.shape[x] for x in range(4)]
    size_data = 28
    alt_index.append(size_data)

    alt_training_set = np.empty(shape=alt_index)
    for label in range(training_set.shape[0]):
        for per in range(training_set.shape[1]):
            for samp in range(training_set.shape[3]):
                alt_training_set[label, per, 0, samp, :] = ar_fit(training_set[label, per, 0, samp, :])
                alt_training_set[label, per, 1, samp, :] = ar_fit(training_set[label, per, 1, samp, :])
    return alt_training_set


def main():
    """For Use."""
    signal = ImportHandData.load_hand_signals()
    ch_list = [1, 2]
    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    training_set_raw = makefullset(signal, labels, ch_list)
    # training_set = training_set_raw
    training_set = alt_training(training_set_raw)
    for chn_it in range(len(ch_list)):
        training_set[:, :, chn_it, :, :] = training_set[:, :, chn_it, :, :] / np.mean(training_set[:, :,
                                                                                      chn_it, :, :])
    train_list = []
    test_list = []
    for person in range(training_set.shape[1]):
        train_split = 0.8
        train_r = int(train_split * training_set.shape[3])
        x_train = training_set[:, person, :, :train_r, :]
        x_test = training_set[:, person, :, :-train_r, :]
        train_list.append(x_train)
        test_list.append(x_test)
    x_train_full = np.stack(train_list, axis=1)
    x_test_full = np.stack(test_list, axis=1)

    print('Data Processed And Loaded In')
    return x_train_full, x_test_full, training_set


if __name__ == '__main__':
    # data_AR = main()
    signal = ImportHandData.load_hand_signals()
    ch_list = [1, 2]
    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    training_set_raw = makefullset(signal, labels, ch_list)
    print(training_set_raw.shape)
