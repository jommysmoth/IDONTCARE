"""Yeah."""
import time

from sklearn.manifold import TSNE
from torch.autograd import Variable
import torch
import loadin
from EMG_RNN import Net
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def evaluate(data_tensor):
    """Yeah."""
    hidden = rnn.init_hidden()

    for i in range(data_tensor.size()[0]):
        output, hidden = rnn(data_tensor[i], hidden)
    return output


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


if __name__ == '__main__':
    time_start = time.time()
    tsne = TSNE(n_components=3, verbose=1, perplexity=100, n_iter=300)

    rnn = torch.load('Trained_Model/trained_model.out')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    training_set, test_set, full_set = loadin.main()
    chn = full_set.shape[2]

    for_tsne = []
    label_list = []

    for label in range(len(labels)):
        for per in range(full_set.shape[1]):
            for ex in range(full_set.shape[3]):
                data = full_set[label, per, :, ex, :]
                input_data = Variable(making_tensor(data, chn))
                output = evaluate(input_data)
                for_tsne.append(output.data[0].numpy())
                label_list.append(label)
    tsne_input = np.stack(for_tsne, axis=0)
    # tsne_input = tsne_input / np.mean(tsne_input)
    label_color = np.array(label_list)
    tsne_embedded = tsne.fit_transform(tsne_input)
    print(tsne_embedded.shape)
    ax.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], tsne_embedded[:, 2], c=label_color)
    ax = plt.gca(projection='3d')
    ax._axis3don = False
    plt.legend(labels)
    plt.show()



