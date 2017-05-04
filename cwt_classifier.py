import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import tensorflow as tf

import frequecy_feature_extraction
import metrics
import tf_helper


class CWTClassifier(object):
    def __init__(self,
                 batch_size,
                 learning_rate,
                 beta1,
                 beta2,
                 require_improvement,
                 seed,
                 num_iterations,
                 input_dim,
                 num_classes,
                 batch_norm,
                 keep_prob,
                 gpu_memory_fraction,
                 train,
                 test,
                 valid,
                 l2_reg,
                 ration_observation,
                 valid_idx,
                 test_idx,
                 hidden_dim=50
                 ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.seed = seed
        self.require_improvement = require_improvement
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.num_classes = num_classes
        self.ration_observation = ration_observation
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = batch_norm
        self.l2_reg = l2_reg
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        self.keep_prob = keep_prob
        self.train_x, self.train_y = train[0], train[1]
        self.valid_x, self.valid_y = valid[0], valid[1]
        self.test_x, self.test_y = test[0], test[1]
        self.num_examples = self.train_x.shape[0]

        self._build_graph()
        self.train_cost, self.train_acc = [], []
        self.validation_cost, self.validation_acc = [], []
        self.valid_idx = valid_idx
        self.test_idx = test_idx

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, self.input_dim], name='x')
            self.y = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
            self.is_training = tf.placeholder(tf.bool)
            self._objective()
            self.saver = tf.train.Saver()
            self.session = tf.Session(config=self.config)
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/mlp_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)
            self.merged = tf.summary.merge_all()

    def _objective(self):
        self.y_logits, self.y_pred_cls, self.cost = self.build_model()
        tf.summary.scalar('cost', self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1,
                                                beta2=self.beta2).minimize(self.cost)

    def train_neural_network(self):
        print_training = "Training MLP: batch_norm:{}, keep_prob:{}".format(self.batch_norm, self.keep_prob)
        print(print_training)
        logging.debug(print_training)
        self.session.run(tf.global_variables_initializer())
        best_validation_accuracy = 0
        last_improvement = 0

        start_time = time.time()
        idx = 0
        epochs = 0
        for i in range(self.num_iterations):
            # Batch Training
            j = self.get_last_batch_index(self.num_examples, idx, self.batch_size)
            x_batch, y_batch = self.train_x[idx:j, :], self.train_y[idx:j, :]
            # TODO simplify batch processing
            if j == self.num_examples:
                epochs += 1
                idx = 0
                is_epoch = True
            else:
                is_epoch = False
                idx = j

            summary, train_loss, train_y_pred_cls, _ = self.session.run(
                [self.merged, self.cost, self.y_pred_cls, self.optimizer],
                feed_dict={self.x: x_batch, self.y: y_batch,
                           self.is_training: True})

            train_cls_true = metrics.convert_labels_to_cls(y_batch)
            train_correct = (train_y_pred_cls == train_cls_true)
            train_acc, _ = metrics.cls_accuracy(train_correct)
            self.train_cost.append(train_loss)
            self.train_acc.append(train_acc)
            self.train_writer.add_summary(summary, i)

            # Calculate the accuracy
            valid_correct, _, valid_cost = self.predict_cls(images=self.valid_x,
                                                            labels=self.valid_y,
                                                            cls_true=metrics.convert_labels_to_cls(self.valid_y))
            validation_acc, _ = metrics.cls_accuracy(valid_correct)
            self.validation_acc.append(validation_acc)
            self.validation_cost.append(valid_cost)

            if is_epoch or (i == (self.num_iterations - 1)):

                if validation_acc > best_validation_accuracy:
                    # Save  Best Perfoming all variables of the TensorFlow graph to file.
                    self.saver.save(sess=self.session, save_path=self.save_path)
                    # update best validation accuracy
                    best_validation_accuracy = validation_acc
                    last_improvement = i
                    improved_str = '*'
                else:
                    improved_str = ''

                print_opt = "Epoch: {}, Training Loss:{}, Acc: {}, " \
                            " Validation Loss:{}, Acc:{} {}".format(epochs, train_loss, train_acc, valid_cost,
                                                                    validation_acc, improved_str)
                print(print_opt)
                logging.debug(print_opt)
            if i - last_improvement > self.require_improvement:
                print_impro = "No improvement found in a while, stopping optimization."
                print(print_impro)
                logging.debug(print_impro)
                # Break out from the for-loop.
                break
                # Ending time.
        end_time = time.time()
        time_dif = end_time - start_time
        print_time = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(print_time)
        logging.debug(print_time)
        return last_improvement, epochs

    def predict_cls(self, images, labels, cls_true):
        num_images = len(images)
        cls_pred = np.zeros(shape=num_images, dtype=np.int)
        idx = 0
        total_loss = 0.0
        num_batches = num_images / self.batch_size
        while idx < num_images:
            # The ending index for the next batch is denoted j.
            j = min(idx + self.batch_size, num_images)
            batch_images = images[idx:j, :]
            batch_labels = labels[idx:j, :]
            feed_dict = {self.x: batch_images,
                         self.y: batch_labels, self.is_training: False}
            cls_pred[idx:j], batch_cost = self.session.run([self.y_pred_cls, self.cost],
                                                           feed_dict=feed_dict)
            total_loss += batch_cost
            idx = j
        # Create a boolean array whether each image is correctly classified.
        correct = (cls_true == cls_pred)
        return correct, cls_pred, total_loss / num_batches

    def build_model(self):
        with tf.variable_scope("y_classifier"):
            w_h1, b_h1 = tf_helper.create_nn_weights('y_h1', 'infer', [self.input_dim, self.hidden_dim])
            w_h2, b_h2 = tf_helper.create_nn_weights('y_h2', 'infer', [self.hidden_dim, self.hidden_dim])
            w_y, b_y = tf_helper.create_nn_weights('y_fully_connected', 'infer', [self.hidden_dim, self.num_classes])

            h1 = tf_helper.dropout_normalised_mlp(self.x, w_h1, b_h1, is_training=self.is_training,
                                                  batch_norm=self.batch_norm, keep_prob=self.keep_prob)
            h2 = tf_helper.dropout_normalised_mlp(h1, w_h2, b_h2, is_training=self.is_training,
                                                  batch_norm=self.batch_norm, keep_prob=self.keep_prob)
            class_weight = tf.constant([self.ration_observation, 1.0 - self.ration_observation])
            logits = tf_helper.mlp_neuron(h2, w_y, b_y, activation=False)
            weighted_logits = tf.multiply(logits, class_weight)  # shape [batch_size, 2]
            y_deci_stats = tf.nn.softmax(weighted_logits)
            y_pred_cls = tf.argmax(y_deci_stats, axis=1)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits, labels=self.y)
            cost = tf.reduce_mean(cross_entropy) + self.l2_loss(self.l2_reg)
        return weighted_logits, y_pred_cls, cost

    def l2_loss(self, scale):
        l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        return l2 * scale

    def train_test(self):
        best_iteration, epochs = self.train_neural_network()
        metrics.plot_cost(training=self.train_cost, validation=self.validation_cost,
                          name="Cross_Entropy_Loss",
                          epochs=epochs,
                          best_iteration=best_iteration)
        metrics.plot_cost(training=self.train_acc, validation=self.validation_acc,
                          name="Accuracy",
                          epochs=epochs,
                          best_iteration=best_iteration)
        self.saver.restore(sess=self.session, save_path=self.save_path)
        correct, cls_pred, test_loss = self.predict_cls(images=self.test_x,
                                                        labels=self.test_y,
                                                        cls_true=metrics.convert_labels_to_cls(self.test_y))

        feed_dict = {self.x: self.test_x,
                     self.y: self.test_y, self.is_training: False}
        logits = self.session.run(self.y_logits, feed_dict=feed_dict)
        print("Test Loss:{}".format(test_loss))
        test_acc = metrics.print_test_accuracy(correct, cls_pred, self.test_y, logging)
        test_auc = metrics.plot_roc(logits, self.test_y, self.num_classes,
                                    name='CWT_{}_{}'.format(self.valid_idx, self.test_idx))
        test_f1_score = metrics.calculate_f1_score(y_true=metrics.convert_labels_to_cls(self.test_y), y_pred=cls_pred)
        return test_acc, test_auc, test_f1_score

    @staticmethod
    def get_last_batch_index(input_size, idx, batch_size):
        # print("input_size:{}, idx:{}, batch_size:{}".format(input_size, idx, batch_size))
        if idx == input_size:
            idx = 0
        j = min(idx + batch_size, input_size)
        return j


def reshape_data(data):
    return data.reshape(data.shape[0], data.shape[1] * data.shape[2])


def encode_label(label):
    label_column_1 = label[:, 1]
    print("label:{}".format(label_column_1.shape))
    idx_range = np.arange(0, label.shape[0])
    label_encoded = np.array([[0, 0], ] * label.shape[0])
    lab_percent = sum(label_column_1) / label.shape[0]
    print("label_1 percent:{}".format(lab_percent))
    for i in idx_range:
        if label_column_1[i] == 0:
            label_encoded[i] = [1, 0]
        else:
            label_encoded[i] = [0, 1]
    return label_encoded, lab_percent


if __name__ == '__main__':
    # TODO test all labels classification
    # TODO implement dropout and batch_norm
    args = sys.argv[1:]
    args_print = "args:{}".format(args)
    print(args_print)
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
        'keep_prob': 0.9
    }
    np.random.seed(FLAGS['seed'])
    subjects = 32
    cross_valid_acc = []
    cross_valid_auc = []
    cross_valid_f1_score = []
    subj_idx = np.arange(start=1, step=1, stop=subjects + 1)
    p = np.array([1 / subjects] * subjects)
    cwt = frequecy_feature_extraction.FrequencyFeatureExtraction()
    held_out_obs = np.random.choice(subj_idx, (2, 16), replace=False, p=p)
    print("held_our_obs:{}, shape:{}".format(held_out_obs, held_out_obs.shape))
    for cross_valid_it in np.arange(held_out_obs.shape[1]):
        valid_idx, test_idx = held_out_obs[0, cross_valid_it], held_out_obs[1, cross_valid_it]
        log_file = 'mlp_classifier_v{}__t{}.log'.format(valid_idx, test_idx)
        logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
        idx_cross = "valid_idx:{}, test_idx:{}".format(valid_idx, test_idx)
        logging.debug(idx_cross)
        logging.debug(held_out_obs)
        print(idx_cross)
        cwt.load_features(valid_idx=valid_idx, test_idx=test_idx)
        logging.debug(args_print)

        train_label, train_ration = encode_label(np.load('MHCTW/train_label.npy'))
        logging.debug("train_obs:{}".format(train_ration))
        train_data = [np.load('MHCTW/train_data.npy'), train_label]
        valid_label, _ = encode_label(np.load('MHCTW/valid_label.npy'))
        valid_data = [np.load('MHCTW/valid_data.npy'), valid_label]
        test_label, _ = encode_label(np.load('MHCTW/test_label.npy'))
        test_data = [np.load('MHCTW/test_data.npy'), test_label]

        data_infor_print = "test:{}, valid:{}, train:{}".format(test_data[0].shape, valid_data[0].shape,
                                                                train_data[0].shape)
        print(data_infor_print)
        logging.debug(data_infor_print)
        cwt_classifier = CWTClassifier(batch_size=FLAGS['batch_size'], learning_rate=FLAGS['learning_rate'],
                                       beta1=FLAGS['beta1'], beta2=FLAGS['beta2'],
                                       require_improvement=FLAGS['require_improvement'], seed=FLAGS['seed'],
                                       num_iterations=FLAGS['num_iterations'],
                                       num_classes=train_data[1].shape[1], input_dim=train_data[0].shape[1],
                                       batch_norm=True,
                                       gpu_memory_fraction=vm, keep_prob=FLAGS['keep_prob'], train=train_data,
                                       test=test_data,
                                       valid=valid_data, l2_reg=FLAGS['l2_reg'], valid_idx=valid_idx, test_idx=test_idx,
                                       ration_observation=0.5)

        with cwt_classifier.session:
            acc, auc, f1_score = cwt_classifier.train_test()
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
    np.save('cwt_cross_vald_auc', cross_valid_auc)
    np.save('cwt_cross_vald_acc', cross_valid_acc)
    np.save('cross_valid_f1_score', cross_valid_f1_score)
    metrics.plot_line(cross_valid_acc, name="CWT Cross Valid ACC")
    metrics.plot_line(cross_valid_auc, name="CWT Cross Valid_AUC")
    metrics.plot_line(cross_valid_f1_score, name="CWT Cross Valid F1")
