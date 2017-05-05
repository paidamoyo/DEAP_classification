import os

import matplotlib as mpl
import numpy as np

mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
import matplotlib.cm as cm
from wavelets import WaveletAnalysis, Ricker
from  pca_features import PCAAnalysis
from utils.load_features import LoadData


class FrequencyFeatureExtraction(object):
    def __init__(self, compute_cwt=False):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.subjects = 32
        self.num_labels = 4
        self.fs = 128
        np.random.seed(31415)
        self.channels = 32
        self.subject_1 = loadmat(os.path.abspath(os.path.join(self.dir_path, '', "DEAP_s/s_{}.mat".format(1))))['data']
        print("subject_1:{}".format(self.subject_1.shape))
        self.folder = 'CWT'
        if not os.listdir(self.folder) or compute_cwt:
            print("extracting frequency features:")
            self.extract_cwt_features()

    def ricket_cwt(self, data):
        widths = np.arange(1, 21)
        cwtmatr = signal.cwt(data, signal.ricker, widths)
        return cwtmatr, self.get_max_freq(cwtmatr)

    def extract_cwt_features(self):
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
                channels_max_freq = []
                s_label_obs = s_label[obs, :]
                for channel in np.arange(self.channels):
                    cwtmatr, max_freq = self.ricket_cwt(s_data[obs, channel, :])
                    channels_max_freq.append(max_freq)
                subject_data.append(channels_max_freq)
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
        print('CWT feature extraction complete')

    def pca_transform(self, observation_freq, n_components):
        pca = PCAAnalysis()
        pca_trans = pca.pca_components(observation_freq, n_components).transform(observation_freq)
        # print("pca_trans:{}{}".format(pca_trans.shape, pca_trans))
        pca_trans = np.reshape(pca_trans, newshape=(self.channels * n_components))
        return pca_trans

    def plot_ricket_transform(self, trial):
        ctwmatr, _ = self.ricket_cwt(data=self.subject_1[trial, 1, :])
        name = "subject1 channel1 trial{} ricket tranform".format(trial)
        plt.figure()
        plt.imshow(ctwmatr, cmap=cm.spectral_r, aspect='auto',
                   vmax=abs(ctwmatr).max(), vmin=-abs(ctwmatr).max())
        plt.savefig(name)

    def wavelet_clean(self, trial):
        origin = 'lower'
        name = "subject1 channel1 trial{} ricket wavelet transform".format(trial)
        s_data_subject = self.subject_1[trial, 1, :]
        wa, max_freq = self.clean_tranform(s_data_subject)
        # wavelet power spectrum
        power = wa.wavelet_power
        # associated time vector
        t = wa.time
        # scales
        scales = wa.scales[::-1]
        fig, ax = plt.subplots()
        T, S = np.meshgrid(t, scales)
        print("power:{}, T:{} S:{}".format(power.shape, T.shape, S.shape))
        CS = plt.contourf(T, S, power, 100,
                          origin=origin)
        # Make a colorbar for the ContourSet returned by the contourf call.
        plt.colorbar(CS)
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        plt.title(name)
        fig.savefig(name)

        plt.figure()
        plt.plot(np.arange(start=0, stop=63, step=1 / 128), max_freq)
        plt.ylabel('Coefficient')
        plt.xlabel('Time [sec]')
        plt.xlim([0, 65])
        plt.title(name + ' max coefficient')
        plt.savefig(name + ' max coefficient')

    def clean_tranform(self, s_data_subject):
        wa = WaveletAnalysis(data=s_data_subject, wavelet=Ricker(), dt=1 / 128)
        max_freq = self.get_max_freq(wa.wavelet_power)
        return wa, max_freq

    def get_max_freq(self, ctwmatr):
        # np.argmax(ctwmatr, axis=0)
        return np.max(ctwmatr, axis=0)

    def plot_spectrogram(self, trial):
        s_data_subject = self.subject_1[trial, 1, :]
        name = "subject1 channel1 trial{} spectrogram".format(trial)
        plt.figure()
        f, t, Sxx = signal.spectrogram(x=s_data_subject, fs=self.fs)
        plt.figure()
        plt.pcolormesh(t, f, Sxx)
        print("Spectrogram:{}, time:{}, f:{}".format(Sxx.shape, t.shape, f.shape))
        plt.ylabel('Frequency [Hz]')
        plt.ylim([0, 61])
        plt.xlim([0, 61])
        plt.xlabel('Time [sec]')
        plt.show()
        plt.title(name)
        plt.savefig(name)

    def plot_power_spectrum(self, trial):
        name = "subject1 channel1 trial{} power_spectrum".format(trial)
        s_data_subject = self.subject_1[trial, 1, :]
        Pxx_den, f = self.power_spectrum(s_data_subject)
        plt.figure()
        plt.semilogy(f, Pxx_den)
        plt.ylim([0.5e-3, 1e1])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title(name)
        plt.savefig(name)

    def power_spectrum(self, s_data_subject):
        f, Pxx_den = signal.welch(s_data_subject, self.fs, nperseg=1024)
        print("power_spectrum:{}".format(Pxx_den.shape))
        return Pxx_den, f

    def load_features(self, valid_idx, test_idx):
        load = LoadData(folder=self.folder)
        return load.load_features(test_idx=test_idx, valid_idx=valid_idx)


if __name__ == '__main__':
    np.random.seed(31415)
    cwt = FrequencyFeatureExtraction()
    # subject 1 trial 1
    cwt.plot_spectrogram(trial=1)
    cwt.plot_power_spectrum(trial=1)
    cwt.wavelet_clean(trial=1)
    # cwt.plot_ricket_transform(trial=1)
    # subject 1 trial 9
    cwt.plot_spectrogram(trial=9)
    cwt.plot_power_spectrum(trial=9)
    cwt.wavelet_clean(trial=9)
    # cwt.plot_ricket_transform(trial=9)
    cwt.load_features(valid_idx=1, test_idx=2)
