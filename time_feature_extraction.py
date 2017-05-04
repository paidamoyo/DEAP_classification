import os

import numpy as np
from scipy.io import loadmat


class TimeFeatureExtraction(object):
    def __init__(self, ):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.subjects = 32
        self.num_labels = 4

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
            print("path:{}".format(path))
            s = loadmat(path)
            s_label = s['label']
            s_data = s['data']
            print("data:{}, label:{}".format(s_data.shape, s_label.shape))
            for obs in np.arange(s_data.shape[0]):
                if subj == valid_idx:
                    valid_data.append(s_data[obs, :, :])
                    valid_lab.append(s_label[obs, :])
                elif subj == test_idx:
                    test_data.append(s_data[obs, :, :])
                    test_lab.append(s_label[obs, :])
                else:
                    train_data.append(s_data[obs, :, :])
                    train_lab.append(s_label[obs, :])

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

        np.save('CONV/{}_label'.format(name), label)
        np.save('CONV/{}_data'.format(name), data)
        return data, label


if __name__ == '__main__':
    np.random.seed(31415)
    time = TimeFeatureExtraction()
    data = time.extract_features(valid_idx=1, test_idx=2)
