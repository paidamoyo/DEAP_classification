import os

import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import matplotlib.cm as cm


class MHFeatureExtraction(object):
    def __init__(self, ):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.subjects = 32
        self.num_labels = 4
        self.channels = 32

    def daubcwt(self, data):
        widths = np.arange(1, 21)
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        return cwtmatr, self.get_max_freq(cwtmatr)

    def moments(self, data):
        mean = np.mean(data, 0)
        constant = 1e-10
        std = np.std(data, 0) + constant
        print_moments = "mean {}, std:{}".format(mean.shape, std.shape)
        print(print_moments)
        return mean, std

    def transform_inputs(self, components, data):
        return np.dot(data, components.T)

    def extract_features(self, test_idx, valid_idx):
        train_data = []
        train_lab = []
        valid_data = []
        valid_lab = []
        test_data = []
        test_lab = []

        print("valid_idx:{}, test_idx:{}".format(valid_idx, test_idx))
        for subj in np.arange(start=1, stop=self.subjects + 1, step=1):
            print("subject:{}".format(subj))
            file = "DEAP_s/s_{}.mat".format(subj)
            path = os.path.abspath(os.path.join(self.dir_path, '', file))
            s = loadmat(path)
            s_label = s['label']
            s_data = s['data']
            print("data:{}, label:{}".format(s_data.shape, s_label.shape))
            for obs in np.arange(s_data.shape[0]):
                s_label_obs = s_label[obs, :]
                for channel in np.arange(self.channels):
                    _, maxfreq = self.daubcwt(s_data[obs, channel, :])
                    if subj == valid_idx:
                        valid_data.append(maxfreq)
                        valid_lab.append(s_label_obs)
                    elif subj == test_idx:
                        test_data.append(maxfreq)
                        test_lab.append(s_label_obs)
                    else:
                        train_data.append(maxfreq)
                        train_lab.append(s_label_obs)

        data = {'train': [np.array(train_data), np.array(train_lab)],
                'valid': [np.array(valid_data), np.array(valid_lab)],
                'test': [np.array(test_data), np.array(test_lab)]}

        self.shuffle_obs(data['train'], name='train')
        self.shuffle_obs(data['valid'], name='valid')
        self.shuffle_obs(data['test'], name='test')
        return data

    def shuffle_obs(self, observations, name):
        signal = observations[0]
        lab = observations[1]
        print('{} cwt_signal:{}, labels:{}'.format(name, signal.shape, lab.shape))

        trials = signal.shape[0]
        idx_range = np.arange(trials)
        np.random.shuffle(idx_range)
        data = signal[idx_range]
        label = lab[idx_range]
        label_file = os.path.abspath(os.path.join(self.dir_path, '', 'MHCTW/{}_label'.format(name)))
        data_file = os.path.abspath(os.path.join(self.dir_path, '', 'MHCTW/{}_data'.format(name)))
        np.save(label_file, label)
        np.save(data_file, data)
        return data, label

    def plot_spectrum(self, ctwmatr, max_freq, name):

        plt.figure()
        plt.imshow(ctwmatr, cmap=cm.spectral_r, aspect='auto',
                   vmax=abs(ctwmatr).max(), vmin=-abs(ctwmatr).max())

        plt.show()
        title = 'signal_{}'.format(name)
        path = os.path.abspath(os.path.join(self.dir_path, '', title))
        plt.title(title)
        plt.savefig(path)
        print("max_freq:{}".format(max_freq.shape))
        plt.figure()
        plt.scatter(np.arange(start=1, stop=8064 + 1, step=1), max_freq)
        file = "max_freq{}".format(name)
        plt.title(file)
        path = os.path.abspath(os.path.join(self.dir_path, '', file))
        plt.savefig(path)

    def get_max_freq(self, ctwmatr):
        return np.max(ctwmatr, axis=0)


def plot_examples(signal):
    s = loadmat(os.path.abspath(os.path.join(cwt.dir_path, '', "DEAP_s/s_{}.mat".format(signal))))
    s_data = s['data']
    coeff, max_freeq = cwt.daubcwt(s_data[signal, 1, :])
    print("coeff:{}".format(coeff.shape))
    cwt.plot_spectrum(coeff, max_freeq, "s_{}_spectral".format(signal))


if __name__ == '__main__':
    np.random.seed(31415)
    cwt = MHFeatureExtraction()
    # cwt.extract_features(valid_idx=1, test_idx=2)
    plot_examples(signal=1)
    plot_examples(signal=9)
