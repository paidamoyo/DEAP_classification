import os

import matplotlib as mpl
import numpy as np

mpl.use('Agg')


class LoadData(object):
    def __init__(self, folder):
        self.subjects = 32
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.folder = folder

    def load_features(self, test_idx, valid_idx):
        print("loading features from:{}".format(self.folder))
        train_data = []
        train_lab = []
        valid_data = []
        valid_lab = []
        test_data = []
        test_lab = []

        print("valid_idx:{}, test_idx:{}".format(valid_idx, test_idx))
        for subj in np.arange(start=1, stop=self.subjects + 1, step=1):
            print("subject:{}".format(subj))
            data_file = os.path.abspath(os.path.join(self.dir_path, '.', "{}/s_{}_data.npy".format(self.folder, subj)))
            label_file = os.path.abspath(
                os.path.join(self.dir_path, '.', "{}/s_{}_label.npy".format(self.folder, subj)))
            s_label = np.load(label_file)
            s_data = np.load(data_file)
            print("data:{}, label:{}".format(s_data.shape, s_label.shape))
            for obs in np.arange(s_data.shape[0]):
                s_label_obs = s_label[obs, :]
                s_data_obs = s_data[obs, :, :]
                if subj == valid_idx:
                    valid_data.append(s_data_obs)
                    valid_lab.append(s_label_obs)
                elif subj == test_idx:
                    test_data.append(s_data_obs)
                    test_lab.append(s_label_obs)
                else:
                    train_data.append(s_data_obs)
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
        folder = 'CONV/'
        label_file = os.path.abspath(os.path.join(self.dir_path, '.', '{}{}_label'.format(folder, name)))
        data_file = os.path.abspath(os.path.join(self.dir_path, '.', '{}{}_data'.format(folder, name)))
        np.save(label_file, label)
        np.save(data_file, data)
        return data, label
