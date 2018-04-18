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
import loadin
import fake_boi
import matplotlib.pyplot as plt


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
        self.softmax = nn.LogSoftmax(dim=1)

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
        return Variable(torch.randn(1, self.hidden_size))


def making_tensor(data, chn):
    """
    Trying.

    Kek
    data - chn - data points
    """
    tensor = torch.zeros(data.shape[1], 1, chn)
    for ch_am in range(chn):
        for li, val in enumerate(data[ch_am, :]):
            tensor[li][0][ch_am] = val
    return tensor


def making_train_tensor(data, chn):
    """
    Trying.

    Kek
    data - chn - data points
    """
    tensor = torch.randn(chn, 1, data.shape[2])
    for ch_am in range(chn):
        for li, val in enumerate(data[ch_am, :, :]):
            tensor[0][li] = val
    return tensor


def rand():
    """
    Pick random train sample and label.

    Might not be necessary

    label is int now, so picked randomly, can be set to string
    with addtional functions for computation
    use dict for different input size, np array won't work
    """
    label = np.random.randint(0, n_categories)
    data = training_set[label,
                        np.random.randint(0, training_per - 1),
                        :,
                        np.random.randint(0, training_sets - 1),
                        :]
    label_tensor = Variable(torch.LongTensor([label]))
    data_tensor = Variable(making_tensor(data, chn))
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
    return labels[label], label


if __name__ == '__main__':

    all_losses = []
    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    # labels = ['Even', 'Odd', 'Rod', 'Smod', 'Dod']
    learning_rate = 0.001
    total_loss = 0
    # numbers of different movement types

    fake_samples = 1000
    fake_length = 50
    fake_chn = 10
    fake_labels = len(labels)
    # training_set, test_set = fake_boi.main(fake_samples, fake_length, fake_chn, fake_labels)
    bat_size = 20

    training_set, test_set, full_set = loadin.main()
    n_categories = training_set.shape[0]
    training_sets = training_set.shape[3]
    training_per = training_set.shape[1]
    chn = training_set.shape[2]
    data_len = training_set.shape[4]
    print(data_len)
    start = time.time()

    for n_hidden in [8, 16, 32, 64, 128, 256]:
        all_losses = []
        total_loss = 0
        rnn = Net(chn, n_hidden, n_categories)
        # rnn.cuda()
        criterion = nn.NLLLoss()

        set_size = training_sets * training_per

        n_iter = 250000
        print_every = n_iter / 10
        plot_every = n_iter / 50
        plot_loss = 0

        for iter in range(1, n_iter + 1):

            label, data, label_tensor, data_tensor = rand()
            output, loss = train(label_tensor, data_tensor)
            total_loss += loss
            plot_loss += loss

            if iter % print_every == 0:
                guess, guess_i = labelout(output)
                correct = '✓' if guess_i == label else '✗ (%s)' % labels[label]
                print('%d %d%% (%s) %.4f / %s %s' % (iter,
                                                     iter / n_iter * 100,
                                                     timesince(start),
                                                     total_loss / print_every,
                                                     guess,
                                                     correct))
                total_loss = 0
            if iter % plot_every == 0:
                all_losses.append(plot_loss / plot_every)
                plot_loss = 0

        for label in range(len(labels)):
            suc = 0
            attempts = 0
            for per in range(test_set.shape[1]):
                for ex in range(test_set.shape[3]):
                    data = test_set[label, per, :, ex, :]
                    input_data = Variable(making_tensor(data, chn))
                    output = evaluate(input_data)
                    guess, guess_i = labelout(output)
                    if guess_i == label:
                        suc += 1
                    attempts += 1
            accuracy = (suc / attempts) * 100
            print('\n\n Accuracy of label %s is: %f' % (labels[label],
                                                        accuracy))
        plt.title('Amount of samples %i' % int(training_sets * training_per))
        if n_hidden == 32:
            col = 'r.-'
        elif n_hidden == 64:
            col = 'b.-'
        elif n_hidden == 128:
            col = 'g.-'
        elif n_hidden == 256:
            col = 'k.-'
        elif n_hidden == 8:
            col = 'y.-'
        elif n_hidden == 16:
            col = 'g.-'
        plt.plot(all_losses, col)
        path = 'Trained_Model/trained_model_%i_hl.out' % n_hidden
        torch.save(rnn, path)
    plt.show()

