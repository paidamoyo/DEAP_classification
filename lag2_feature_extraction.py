import os

import matplotlib as mpl
import numpy as np
from scipy.io import loadmat

mpl.use('Agg')
import matplotlib.pyplot as plt


def view_s1():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', "DEAP_s/s_1.mat"))
    print("path:{}".format(path))
    s1 = loadmat(path)
    s1_label = s1['label']
    s1_data = s1['data']
    print("s1.label:{}, s1.data:{}".format(s1_label.shape, s1_data.shape))
    s1_obs1_chan1 = s1_data[1, 1, :]
    label_obs1 = s1_label[1, :]
    s1_obs9_chan1 = s1_data[9, 1, :]
    label_obs9 = s1_label[9, :]
    print("s1_obs1_chan1:{}, label:{}".format(s1_obs1_chan1.shape, label_obs1))
    print("s1_obs9_chan1:{}, label:{}".format(s1_obs1_chan1.shape, label_obs9))
    plot_single_channel(signal=s1_obs1_chan1, name='s1_obs1_chan_1')
    plot_single_channel(signal=s1_obs9_chan1, name='s1_obs9_chan1')

    idx = np.arange(int(s1_obs1_chan1.shape[0] - 1))
    print("idx:{}".format(idx))
    s1_obs1_chan1_lag2 = lag_2(s1_obs1_chan1, idx)
    s1_obs9_chan1_lag2 = lag_2(s1_obs9_chan1, idx)
    plot_lag(s1_obs1_chan1_lag2, name='s1_obs1_chan1_lag2')
    plot_lag(s1_obs9_chan1_lag2, name='s1_obs9_chan1_lag2')


def plot_single_channel(signal, name):
    plt.figure()
    plt.plot(signal)
    plt.xlabel('time')
    plt.ylabel('signal')
    title = 'signal_{}'.format(name)
    plt.title(title)
    plt.savefig(title)


def lag_2(signal, idx):
    output = np.array([[0, 0], ] * len(idx))
    for i in idx:
        output[i] = [signal[i], signal[i + 1]]
    return output


def plot_lag(signal, name):
    plt.figure()
    plt.scatter(signal[:, 0], signal[:, 1])
    plt.xlabel('pt1')
    plt.ylabel('pt2')
    title = 'lag_2_{}'.format(name)
    plt.title(title)
    plt.savefig(title)


def extract_features():
    train_data = []
    train_lab = []
    valid_data = []
    valid_lab = []
    test_data = []
    test_lab = []

    num_trials = 40
    p = np.array([1 / num_trials] * num_trials)
    a = np.arange(num_trials)
    held_out_obs = np.random.choice(a, (2, 5), replace=False, p=p)
    subjects = np.arange(start=1, stop=33, step=1)
    held_out_subj = np.random.choice(subjects, (2, 15), replace=False, p=np.array([1 / 32] * 32))
    valid_obs_idx = held_out_obs[0, :]
    valid_subj_idx = held_out_subj[0, :]
    test_obs_idx = held_out_obs[1, :]
    test_subj_idx = held_out_subj[1, :]
    print("valid_obs_idx:{}, test_obs_idx:{}".format(valid_obs_idx, test_obs_idx))
    print("valid_sub_idx:{}, test_subj_idx:{}".format(valid_subj_idx, test_subj_idx))

    for subj in subjects:
        print("subject:{}".format(subj))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = "DEAP_s/s_{}.mat".format(subj)
        path = os.path.abspath(os.path.join(dir_path, '', file))
        print("path:{}".format(path))
        s = loadmat(path)
        s_label = s['label']
        s_data = s['data']
        idx = np.arange(int(s_data.shape[2] - 1))
        for obs in np.arange(s_data.shape[0]):
            for chan in np.arange(s_data.shape[1]):
                if obs in valid_obs_idx and subj in valid_subj_idx:
                    valid_data.append(lag_2(s_data[obs, chan, :], idx))
                    valid_lab.append(s_label[obs, :])
                elif obs in test_obs_idx and subj in test_subj_idx:
                    test_data.append(lag_2(s_data[obs, chan, :], idx))
                    test_lab.append(s_label[obs, :])
                else:
                    train_data.append(lag_2(s_data[obs, chan, :], idx))
                    train_lab.append(s_label[obs, :])

    return {
        'train': [np.array(train_data), np.array(train_lab)],
        'valid': [np.array(valid_data), np.array(valid_lab)],
        'test': [np.array(test_data), np.array(test_lab)]
    }


def shuffle_obs(observations, name):
    signal = observations[0]
    lab = observations[1]
    print('{} lag2_sig:{}, labels:{}'.format(name, signal.shape, lab.shape))

    idx_range = np.arange(signal.shape[0])
    print("idx_range:{}".format(idx_range))
    np.random.shuffle(idx_range)
    data = signal[idx_range]
    label = lab[idx_range]

    np.save('{}_label'.format(name), label)
    np.save('{}_data'.format(name), data)
    return data, label


if __name__ == '__main__':
    np.random.seed(31415)
    view_s1()
    data = extract_features()
    train = shuffle_obs(data['train'], name='train')
    valid = shuffle_obs(data['valid'], name='valid')
    test = shuffle_obs(data['test'], name='test')