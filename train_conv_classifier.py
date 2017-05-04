import logging
import sys

import numpy as np

import frequecy_feature_extraction
import metrics
from conv_classifier import ConvClassifier


def reshape_data(data):
    return data.reshape(data.shape[0], data.shape[1] * data.shape[2])


def encode_label(label):
    label_column_1 = label[:, 1]
    print("label:{}".format(label_column_1.shape))
    idx_range = np.arange(0, label.shape[0])
    label_encoded = np.array([[0, 0], ] * label.shape[0])
    obs_ration = sum(label_column_1) / label.shape[0]
    observed_perc_print = "label_1 percent:{}".format(obs_ration)
    print(observed_perc_print)
    logging.debug(observed_perc_print)
    for i in idx_range:
        if label_column_1[i] == 0:
            label_encoded[i] = [1, 0]
        else:
            label_encoded[i] = [0, 1]
    return label_encoded, obs_ration


def swap_axes_data(data):
    swapped_data = np.swapaxes(data, 1, 2)
    print("swapped:{}".format(swapped_data.shape))
    return swapped_data


if __name__ == '__main__':
    # TODO test all labels classification
    # TODO implement dropout and batch_norm

    args = sys.argv[1:]
    args_print = "args:{}".format(args)
    if args:
        vm = float(args[0])
    else:
        vm = 1.0
    print("gpu_memory_fraction:{}".format(vm))
    FLAGS = {
        'num_iterations': 10000,  # should 3000 epochs
        'batch_size': 40,
        'seed': 31415,
        'require_improvement': 1000,
        'learning_rate': 3e-4,
        'beta1': 0.9,
        'beta2': 0.999,
        'l2_reg': 0.001,
        'keep_prob': 0.9,
        'filter_sizes': [5, 5],
        'num_filters': [16, 36],
        'fc_size': 128
    }
    np.random.seed(FLAGS['seed'])
    subjects = 32
    cross_valid_acc = []
    cross_valid_auc = []
    cross_valid_f1_score = []
    subj_idx = np.arange(start=1, step=1, stop=subjects + 1)
    p = np.array([1 / subjects] * subjects)
    # conv_feature = conv_feature_extraction.ConvFeatureExtraction()
    conv_feature = frequecy_feature_extraction.FrequencyFeatureExtraction()
    held_out_obs = np.random.choice(subj_idx, (2, 16), replace=False, p=p)
    print("held_our_obs:{}, shape:{}".format(held_out_obs, held_out_obs.shape))
    for cross_valid_it in np.arange(held_out_obs.shape[1]):
        valid_idx, test_idx = held_out_obs[0, cross_valid_it], held_out_obs[1, cross_valid_it]
        log_file = 'conv_classifier_v{}__t{}.log'.format(valid_idx, test_idx)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
        idx_cross = "valid_idx:{}, test_idx:{}".format(valid_idx, test_idx)
        logging.debug(idx_cross)
        logging.debug(held_out_obs)
        print(idx_cross)
        conv_feature.load_features(valid_idx=valid_idx, test_idx=test_idx)
        print(args)
        logging.debug(args_print)

        train_label, train_ration = encode_label(np.load('CONV/train_label.npy'))
        obs_perc_print = "train_obs:{}".format(train_ration)
        print(obs_perc_print)
        logging.debug(obs_perc_print)
        train_data = [swap_axes_data(np.load('CONV/train_data.npy')), train_label]
        valid_label, _ = encode_label(np.load('CONV/valid_label.npy'))
        valid_data = [swap_axes_data(np.load('CONV/valid_data.npy')), valid_label]
        test_label, _ = encode_label(np.load('CONV/test_label.npy'))
        test_data = [swap_axes_data(np.load('CONV/test_data.npy')), test_label]

        data_infor_print = "test:{}, valid:{}, train:{}".format(test_data[0].shape, valid_data[0].shape,
                                                                train_data[0].shape)
        print(data_infor_print)
        logging.debug(data_infor_print)
        conv = ConvClassifier(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                              beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                              require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                              num_iterations=FLAGS['num_iterations'],
                              num_classes=train_data[1].shape[1],
                              input_dim=[None, train_data[0].shape[1], train_data[0].shape[2]], batch_norm=True,
                              gpu_memory_fraction=vm, keep_prob=FLAGS['keep_prob'], train=train_data,
                              test=test_data,
                              valid=valid_data, l2_reg=FLAGS['l2_reg'], filter_sizes=FLAGS['filter_sizes'],
                              fc_size=FLAGS['fc_size'], num_filters=FLAGS['num_filters'], hidden_dim=100,
                              feature_dim=50, valid_idx=valid_idx, test_idx=test_idx, ration_observation=0.5)

        with conv.session:
            acc, auc, f1_score = conv.train_test()
            cross_valid_acc.append(acc)
            cross_valid_auc.append(auc)
            cross_valid_f1_score.append(f1_score)
    final_results_print = "Results acc:{}, auc:{}, f1_score:{}".format(np.mean(cross_valid_acc),
                                                                       np.mean(cross_valid_auc),
                                                                       np.mean(cross_valid_f1_score))
    print(final_results_print)
    logging.debug(final_results_print)
    logging.debug(cross_valid_acc)
    logging.debug(cross_valid_auc)
    logging.debug(cross_valid_f1_score)
    np.save('conv_cross_vald_auc', cross_valid_auc)
    np.save('conv_cross_vald_acc', cross_valid_acc)
    np.save('conv_cross_vald_acc', cross_valid_f1_score)
    metrics.plot_line(cross_valid_acc, name="Conv Cross Valid ACC")
    metrics.plot_line(cross_valid_auc, name="Conv Cross Valid AUC")
    metrics.plot_line(cross_valid_f1_score, name="Conv Cross Valid F1 Score")
