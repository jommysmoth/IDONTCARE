"""
Using CNN on Spectrum Data.

Set up to create spectrum of input for CNN
"""
# Gather data set into paths and class labels
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import CNN_start as net_mod
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import math


def read_in_data():
    """Testing."""
    paths = []
    labels = []
    for path, subdirs, files in os.walk('DigitData'):
        for name in files:
            paths.append(os.path.join(path, name))
            labels.append(re.split(r'(\d+)', name)[0])
    myset = set(labels)
    num_classes = len(myset)

    # Change class label to one-hot vectors
    le = LabelEncoder()
    # Convert to ints

    le.fit(list(myset))
    int_labels = le.transform(labels)

    df = pd.read_csv(paths[0], header=None, nrows=80000)
    timestep = df.shape[0]
    num_channels = df.shape[1]

    # Initialize x data
    x_data = np.empty((len(paths), timestep, num_channels), dtype=np.float32)
    # Load dat to tensor
    samp_list = []
    for i, path in enumerate(paths):
        data = pd.read_csv(path, header=None, nrows=80000)
        x_data[i, :, :] = data.values
        chn_stack = []
        for chn in range(num_channels):
            spec, f, t, im = plt.specgram(x_data[i, :, 0],
                                          Fs=4000,
                                          NFFT=500)
            im_array = im.get_array()
            # im_array = im_array / np.mean(im_array)
            # plt.imshow(im_array)
            chn_stack.append(im_array)
        # plt.show()
        samp_list.append(np.stack(chn_stack, axis=0))
    data_set = np.stack(samp_list, axis=0)

    # Samples - Channels - Height - Width

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data_set, int_labels, test_size=0.2, random_state=40,
                                                        stratify=int_labels)
    return X_train, X_test, y_train, y_test


def labelout(output):
    """Used for accuracy testing."""
    batchsize = output.size()[0]
    batch_bal_list = []
    for examp in range(batchsize):
        samp = output[examp].data
        val, lab = samp.max(0)
        lab = lab[0]
        batch_bal_list.append(lab)
    return batch_bal_list


def timesince(since):
    """
    Timing Training.

    Used to time the length of time the data
    takes to fully train
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return ' %dm %ds' % (m, s)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = read_in_data()

    batches = 50
    epoch = 20
    channels = X_train.shape[1]
    h = X_train.shape[2]
    w = X_train.shape[3]

    cnn = net_mod.Net(batches, channels, h, w, 15)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(cnn.parameters(),
                          lr=0.001)

    start = time.time()
    exit()
    for run in range(epoch):
        total_loss = 0
        count = 0
        for ins in range(X_train.shape[0]):
            data_start = (ins * batches)
            data_end = data_start + batches - 1
            if data_end >= X_train.shape[0]:
                data_end = X_train.shape[0]
            sample = X_train[data_start:data_end, :, :, :]
            target = y_train[data_start:data_end]

            input = torch.from_numpy(sample)
            input = input.type(torch.FloatTensor)

            input = Variable(input)
            target_tensor = torch.LongTensor(target)
            target_tensor = Variable(target_tensor)

            optimizer.zero_grad()

            output = cnn(input)

            loss = criterion(output, target_tensor)
            loss.backward()
            optimizer.step()

            guess = labelout(output)
            total_loss += loss.data[0]
            count += 1
            if data_end == X_train.shape[0]:
                break
        print(guess, target)
        print(total_loss / count)
        print(timesince(start))
