"""
Recurrent Neural Network.

Setting it up bois
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import loadin
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from RNN import Net


def making_tensor(data, chn, type=None):
    """
    Trying.

    Kek
    data - chn - data points
    """
    tensor = torch.zeros(data.shape[1], 1, chn)
    for ch_am in range(chn):
        for li, val in enumerate(data[ch_am, :]):
            tensor[li][0][ch_am] = val

    if type == 'moving':
        """For raw data only"""
        # rand_size = np.random.randint(int(data.shape[1] / 10), int(data.shape[1] / 5))
        rand_size = 10
        rand_start = np.random.randint(0, data.shape[1] - rand_size)
        rand_end = rand_start + rand_size
        tensor = torch.zeros(rand_size, 1, chn)
        for ch_am in range(chn):
            for li, val in enumerate(data[ch_am, rand_start:rand_end]):
                tensor[li][0][ch_am] = val
    return tensor


def rand(type=None):
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
    data_tensor = Variable(making_tensor(data, chn, type=type))
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


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues):
    """
    Function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':

    all_losses = []
    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    learning_rate = 0.001
    total_loss = 0

    training_set, test_set, full_set = loadin.main()
    n_categories = training_set.shape[0]
    training_sets = training_set.shape[3]
    training_per = training_set.shape[1]
    chn = training_set.shape[2]
    data_len = training_set.shape[4]
    save_model = False
    start = time.time()
    conf_m_list = []
    hidden_list = [8, 16, 32, 64, 128, 256]

    for n_hidden in hidden_list:
        all_losses = []
        total_loss = 0
        rnn = Net(chn, n_hidden, n_categories)
        criterion = nn.NLLLoss()

        set_size = training_sets * training_per

        n_iter = 250
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

        y_pred = []
        y_test = []
        for label in range(len(labels)):
            suc = 0
            attempts = 0
            for per in range(test_set.shape[1]):
                for ex in range(test_set.shape[3]):
                    data = test_set[label, per, :, ex, :]
                    input_data = Variable(making_tensor(data, chn))
                    output = evaluate(input_data)
                    guess, guess_i = labelout(output)
                    y_pred.append(guess_i)
                    y_test.append(label)
                    if guess_i == label:
                        suc += 1
                    attempts += 1
            accuracy = (suc / attempts) * 100
            print('\n\n Accuracy of label %s is: %f' % (labels[label],
                                                        accuracy))
        sample_size = int(training_sets * training_per * len(labels))
        plt.title('Loss after %i interations, and samples %i' % (n_iter, sample_size))
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
        if save_model:
            path = 'Trained_Model/trained_model_%i_hl.out' % n_hidden
            torch.save(rnn, path)
        confm = confusion_matrix(y_test, y_pred)
        conf_m_list.append(confm)
    conf_m_plot = np.stack(conf_m_list, axis=0)
    for plot in range(conf_m_plot.shape[0]):
        title_conf = 'Predicted vs. True for n_hidden = %i' % hidden_list[plot]
        plt.figure()
        plot_confusion_matrix(conf_m_plot[plot, :, :], classes=labels, title=title_conf)
    plt.show()
