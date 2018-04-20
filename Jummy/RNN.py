"""
RNN for EMG data.

Model used for Hand Data
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


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
