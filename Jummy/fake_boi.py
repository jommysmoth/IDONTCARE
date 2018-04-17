"""
Loading in Data.

Raw data for now, to be used in RNN
"""
import numpy as np
import sklearn.preprocessing as sp


def make_fake(samples, length, chn, label_am):
    """I NEED YOU."""
    fake_man = np.empty([label_am, 10, chn, samples, length])
    for label in range(fake_man.shape[0]):
        for per in range(fake_man.shape[1]):
            for samp in range(fake_man.shape[3]):
                for point in range(fake_man.shape[4]):
                    """
                    if label == 0:
                        fake_man[label, per, :, samp, point] = 2 * np.random.randint(0, 2 * point + 1,
                                                                                     size=chn)
                    else:
                        fake_man[label, per, :, samp, point] = 2 * np.random.randint(0, 2 * point + 10,
                                                                                     size=chn)
                        fake_man[label, per, :, samp, point] += 1
                    """
                    fake_man[label, per, :, samp, point] = (label + 1) * np.random.randint(0, point + 1,
                                                                                           size=chn)
    for chn_now in range(chn):
        fake_man[:, :, chn_now, :, :] = fake_man[:, :, chn_now, :, :] / np.mean(fake_man[:, :, chn_now, :, :])
    print('Fake 0')
    return fake_man


def make_fake1(samples, length, chn, label_am):
    """I NEED YOU."""
    fake_man = np.zeros([label_am, 10, chn, samples, length])
    for label in range(fake_man.shape[0]):
        for per in range(fake_man.shape[1]):
            for samp in range(fake_man.shape[3]):
                for point in range(fake_man.shape[4]):
                    fake_man[label, per, :, samp, point] = np.full(chn, point * (label + 1))
                # fake_man[label, per, :, samp, :] = sp.normalize(fake_man[label, per, :, samp, :])
            # fake_man[label, per, :, samp, :] = sp.normalize(fake_man[label, per, :, samp, :])
    print('Fake 1')
    return fake_man


def main(samples, length, chn, label_am):
    """For Use."""
    training_set = make_fake(samples, length, chn, label_am)
    training_set = training_set / np.mean(training_set)
    test_set = make_fake(int(samples / 10), length, chn, label_am)
    test_set = test_set / np.mean(test_set)
    return training_set, test_set
