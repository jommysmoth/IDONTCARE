"""Yeah."""
from pathlib import Path
from sklearn.manifold import TSNE
from torch.autograd import Variable
import torch
import loadin
from EMG_RNN import Net
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pickle
import scipy.interpolate as sp
import time


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


def interpolation(input_dict, hidden_list, int_amount):
    """Create Interpolation frames for tsne."""
    output_dict = {}
    int_val_list = []
    range_list = []
    set_size = input_dict[str(hidden_list[0])].shape[0]
    for ins in range(set_size):
        points = []
        for hl in hidden_list:
            data = input_dict[str(hl)][ins, :]
            points.append(data)
        point_set = np.stack(points, axis=1)
        range_list.append(point_set)
        int_val_list.append(sp.splprep(point_set, k=point_set.shape[1] - 1))
    for step, hl in enumerate(hidden_list):
        for int_val in range(int_amount):
            val_list = []
            ran = (step * (int_amount)) + int_val
            char_in = str(ran)
            if int_val == 0:
                output_dict[char_in] = input_dict[str(hl)]
            else:
                for ins in range(set_size):
                    move = np.linspace(0, 1.05, len(hidden_list) * int_amount)
                    val = sp.splev(move[ran], int_val_list[ins][0])
                    val = [float(x) for x in val]
                    val_list.append(np.array(val))
                data_fill = np.stack(val_list, axis=0)
                output_dict[char_in] = data_fill
    return output_dict


def update(move, input, ob, hs, int_amount, type):
    """Update Moving animation."""
    in_ = str(move)
    time.sleep(1)
    if num_com == 3:
        if move % int_amount == 0:
            val = (move + 1) / int_amount
            lnum = str(hs[int(val)])
            ax.set_title('tSNE of RNN with ' + lnum + ' size for hidden layer', y=1)
        ob._offsets3d = (input[in_][:, 0], input[in_][:, 1], input[in_][:, 2])
    elif num_com == 2:
        if move % int_amount == 0:
            val = (move + 1) / int_amount
            lnum = str(hs[int(val)])
            ax.set_title('tSNE of RNN with ' + lnum + ' size for hidden layer', y=1)
        ob.set_offsets(input[in_])
    return ob,


if __name__ == '__main__':
    overwrite = False
    labels = ['cyl', 'hook', 'tip', 'palm', 'spher', 'lat']
    hidden_list = [8, 16, 32, 64, 128, 256]
    # FFwriter = animation.FFMpegWriter(fps=60, codec='libx264')

    file_des = 'Visual Data/tsne_data.pickle'
    num_com = 2
    if not Path(file_des).is_file() or overwrite:
        tsne = TSNE(n_components=num_com, verbose=1, perplexity=100, n_iter=300)

        training_set, test_set, full_set = loadin.main()
        chn = full_set.shape[2]

        outputdict = {}

        for step, hidden_size in enumerate(hidden_list):
            for_tsne = []
            label_list = []
            rnn = torch.load('Trained_Model/trained_model_' + str(hidden_size) + '_hl.out')
            for label in range(len(labels)):
                for per in range(full_set.shape[1]):
                    for ex in range(full_set.shape[3]):
                        data = full_set[label, per, :, ex, :]
                        input_data = Variable(making_tensor(data, chn))
                        output = evaluate(input_data)
                        ts_fill = output.data[0].numpy()
                        # ts_fill = ts_fill / np.mean(ts_fill)
                        for_tsne.append(ts_fill)
                        label_list.append(label)
            tsne_input = np.stack(for_tsne, axis=0)
            tsne_input = tsne_input / np.mean(tsne_input)
            label_color = np.array(label_list)
            tsne_embedded = tsne.fit_transform(tsne_input)
            outputdict[str(step)] = tsne_embedded
        outputdict['Label Color'] = label_color
        outputdict['Size Set'] = full_set.shape[0]
        with open(file_des, 'wb') as handle:
            pickle.dump(outputdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('TSNE Data Saved')
    else:
        with open(file_des, 'rb') as handle:
            outputdict = pickle.load(handle)
        print('TSNE Data Loaded')
    int_amount = 10
    # int_dict = interpolation(outputdict, hidden_list, int_amount)
    int_dict = outputdict
    if num_com == 3:
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        com = '0'
        ax.set_xlim3d([-15, 15])

        ax.set_ylim3d([-15, 15])

        ax.set_zlim3d([-15, 15])
        ax.set_title('tSNE of RNN with ' + str(hidden_list[0]) + ' size for hidden layer', y=1)

        ob = ax.scatter(int_dict[com][:, 0], int_dict[com][:, 1], int_dict[com][:, 2],
                        c=outputdict['Label Color'])

        ani = animation.FuncAnimation(fig, update, int_amount * len(hidden_list),
                                      fargs=[int_dict, ob, hidden_list, int_amount, num_com], interval=1,
                                      blit=False)
        # ani.save('Howard.mp4', writer=FFwriter)
    elif num_com == 2:
        fig, ax = plt.subplots()
        com = '0'
        ob = ax.scatter(int_dict[com][:, 0], int_dict[com][:, 1], c=outputdict['Label Color'])

        # ani_amount = int_amount * len(hidden_list)

        ani_amount = len(hidden_list)

        ani = animation.FuncAnimation(fig, update, ani_amount,
                                      fargs=[int_dict, ob, hidden_list, int_amount, num_com], interval=1,
                                      blit=False)
        ax.set_ylim([-20, 20])
        ax.set_xlim([-20, 20])

    # plt.axis('off')

    plt.show()
