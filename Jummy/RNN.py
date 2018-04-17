"""
Recurrent Neural Network.

Setting it up bois
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Net(nn.Module):
    """
    Recurrent Neural Network.

    This is the class for the network for training
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializer.

        Creating values for network
        """
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        """
        Moving forward.

        Operations for (input, hidden) to move through the
        network and eventually become the (output, hidden)
        """
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Hidden Tensor Initializer.

        Start hidden tensor full of zeros by
        created size
        """
        return Variable(torch.zeros(1, self.hidden_size).cuda())


def dv_index(val):
    """
    Will need to make less trival for non 1-25.

    for now can be basic.
    """
    return(val)


def making_tensor(data, dv):
    """
    Trying.

    Kek
    """
    tensor = torch.zeros(len(data), 1, 8)
    for li, val in enumerate(data):
        tensor[li][0][:] = val
    return tensor


def rand():
    """
    Pick random train sample and label.

    Might not be necessary

    label is int now, so picked randomly, can be set to string
    with addtional functions for computation
    use dict for different input size, np array won't work
    """
    label = np.random.randint(0, n_categories - 1)
    data = fake_set[label][np.random.randint(0, training_sets - 1)][:]
    label_tensor = Variable(torch.LongTensor([label]).cuda())
    data_tensor = Variable(making_tensor(data, dv).cuda())
    return label, data, label_tensor, data_tensor


def train(label_tensor, data_tensor):
    """
    Training the RNN.

    With input and output (get rid of target)
    """
    hidden = rnn.init_hidden()

    rnn.zero_grad()

    for i in range(data_tensor.size()[0]):
        output, hidden = rnn(data_tensor[i], hidden)

    loss = criterion(output, label_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.data[0]


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


def makingfd(training_sets, dv, n_categories, range_vec):
    """
    Making Fake Data.

    Use amount of training sets and length to make fake data

    say we have 25 distinct values, (add changing length to check)

    incorporate different random categories (make random now, different randoms later)
    """
    full_set = []
    full_set_list = []
    wow_full = []
    for cat in range(n_categories):
        for x in range(training_sets):
            for i in range_vec:
                full_set.append(np.random.randint(0, dv - 10) + cat)
            full_set_list.append(full_set)
        wow_full.append(full_set_list)
    return wow_full


def evaluate(data_tensor):
    """Yeah."""
    hidden = rnn.init_hidden()

    for i in range(data_tensor.size()[0]):
        output, hidden = rnn(data_tensor[i], hidden)

        return output


def labelout(output):
    """Yeah."""
    top_n, top_i = output.data.topk(1)
    label = top_i[0][0]
    return label

if __name__ == '__main__':
    training_sets = 50
    print_every = training_sets / 10
    all_losses = []
    learning_rate = 0.0005
    n_hidden = 128
    total_loss = 0
    # numbers of different movement types
    n_categories = 5
    chn = 8

    range_vec = np.random.randint(0, 30, training_sets)
    fake_set = makingfd(training_sets, dv, n_categories, range_vec)

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    rnn = Net(chn, n_hidden, n_categories)
    rnn.cuda()
    criterion = nn.NLLLoss()

    start = time.time()

    for iter in range(1, training_sets + 1):
        label, data, label_tensor, data_tensor = rand()
        output, loss = train(label_tensor, data_tensor)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timesince(start),
                                         iter, iter / training_sets * 100, loss))

    for i in range(n_categories):
        label, data, label_tensor, data_tensor = rand()
        output = evaluate(data_tensor)
        guess = labelout(output)
        confusion[label][guess] += 1

    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
