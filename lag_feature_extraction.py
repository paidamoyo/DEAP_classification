import os

import matplotlib as mpl
import numpy as np
from scipy.io import loadmat

mpl.use('Agg')
import matplotlib.pyplot as plt
from utils.load_features import LoadData


class LAGFeatureExtraction(object):
    def __init__(self, lag=2, compute_lag=False):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.subjects, self.channels, self.time_stamps = 32, 32, 8064
        self.subject_1_path = os.path.abspath(os.path.join(self.dir_path, '', "DEAP_s/s_1.mat"))
        self.folder = 'LAG'
        self.lag = lag
        np.random.seed(31415)
        if not os.listdir(self.folder) or compute_lag:
            print("extracting lag features:")
            self.extract_lag_features()

    def view_s1(self):
        s1 = loadmat(self.subject_1_path)
        s1_label = s1['label']
        s1_data = s1['data']
        print("s1.label:{}, s1.data:{}".format(s1_label.shape, s1_data.shape))
        s1_obs1_chan1 = s1_data[1, 1, :]
        label_obs1 = s1_label[1, :]
        s1_obs9_chan1 = s1_data[9, 1, :]
        label_obs9 = s1_label[9, :]
        print("s1_obs1_chan1:{}, label:{}".format(s1_obs1_chan1.shape, label_obs1))
        print("s1_obs9_chan1:{}, label:{}".format(s1_obs1_chan1.shape, label_obs9))
        self.plot_single_channel(signal=s1_obs1_chan1, name='s1_obs1_chan_1')
        self.plot_single_channel(signal=s1_obs9_chan1, name='s1_obs9_chan1')

        s1_obs1_chan1_lag2 = self.lag_map(s1_obs1_chan1)
        s1_obs9_chan1_lag2 = self.lag_map(s1_obs9_chan1)
        print("s1_obs1_chan1_lag2:{}".format(s1_obs1_chan1_lag2.shape))
        print("s1_obs9_chan1_lag2:{}".format(s1_obs9_chan1_lag2.shape))
        self.plot_lag(s1_obs1_chan1_lag2, name='s1_obs1_chan1_lag2')
        self.plot_lag(s1_obs9_chan1_lag2, name='s1_obs9_chan1_lag2')

    @staticmethod
    def plot_single_channel(signal, name):
        plt.figure()
        plt.plot(signal)
        plt.xlabel('time')
        plt.ylabel('signal')
        title = 'signal_{}'.format(name)
        plt.title(title)
        plt.savefig(title)

    def lag_map(self, signal):
        idx = np.arange(int(self.time_stamps))
        lag_output = []
        for i in idx:
            max_index = i + self.lag
            if max_index >= self.time_stamps:
                break
            else:
                lag_indexes = np.arange(start=i, stop=max_index, step=1)
            lag_trans = np.array(signal[lag_indexes])
            # print("max_index:{}, lag_indexes:{} lag_trans:{}".format(max_index, lag_indexes, lag_trans.shape))
            lag_output.append(lag_trans)
        return np.array(lag_output)

    @staticmethod
    def plot_lag(signal, name):
        plt.figure()
        plt.scatter(signal[:, 0], signal[:, 1])
        plt.xlabel('pt1')
        plt.ylabel('pt2')
        title = 'lag_2_{}'.format(name)
        plt.title(title)
        plt.savefig(title)

    def extract_lag_features(self):
        for subj in np.arange(start=1, stop=self.subjects + 1, step=1):
            print("subject:{}".format(subj))
            file = "DEAP_s/s_{}.mat".format(subj)
            path = os.path.abspath(os.path.join(self.dir_path, '', file))
            s = loadmat(path)
            s_label = s['label']
            s_data = s['data']
            print("data:{}, label:{}".format(s_data.shape, s_label.shape))
            subject_data = []
            subject_label = []
            for obs in np.arange(s_data.shape[0]):
                channels_lag2 = []
                s_label_obs = s_label[obs, :]
                for channel in np.arange(self.channels):
                    lag_map = self.lag_map(s_data[obs, channel, :])
                    channels_lag2.append(lag_map)
                subject_data.append(channels_lag2)
                subject_label.append(s_label_obs)
            subject_data = np.array(subject_data)
            subject_label = np.array(subject_label)
            print("subject_obs:{}, subject_label{}".format(subject_data.shape, subject_label.shape))
            data_file = os.path.abspath(
                os.path.join(self.dir_path, '.', '{}/{}_data'.format(self.folder, 's_{}'.format(subj))))
            label_file = os.path.abspath(
                os.path.join(self.dir_path, '.', '{}/{}_label'.format(self.folder, 's_{}'.format(subj))))
            np.save(data_file, subject_data)
            np.save(label_file, subject_label)
        print('LAG feature extraction complete')

    def load_features(self, valid_idx, test_idx):
        load = LoadData(folder=self.folder)
        return load.load_features(test_idx=test_idx, valid_idx=valid_idx)


if __name__ == '__main__':
    np.random.seed(31415)
    lag_feature = LAGFeatureExtraction()
    lag_feature.view_s1()
    # lag.extract_lag_features()
