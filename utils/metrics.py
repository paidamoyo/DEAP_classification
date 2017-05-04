import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, auc


def convert_labels_to_cls(labels):
    return np.argmax(labels, axis=1)


def calculate_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')


def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum) / len(correct)
    return acc, correct_sum


def plot_confusion_matrix(cls_pred, labels, logging):
    cls_true = convert_labels_to_cls(labels)
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    logging.debug(cm)
    plt.matshow(cm)


def print_test_accuracy(correct, cls_pred, labels, logging):
    acc, correct_sum = cls_accuracy(correct)
    num_images = len(correct)
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_images))
    logging.debug(msg.format(acc, correct_sum, num_images))

    print("Confusion Matrix:")
    logging.debug("Confusion Matrix:")

    plot_confusion_matrix(cls_pred=cls_pred, labels=labels, logging=logging)
    return acc


def plot_roc(logits, y_true, n_classes, name):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    micro_auc = roc_auc['micro']
    plt.plot(fpr['micro'], tpr['micro'], color='darkorange', label='ROC curve (area = {})'.format(micro_auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    title = 'ROC of ' + name + ' model'
    plt.title(title)
    plt.legend(loc="lower right")
    save_path = name + "ROC"
    plt.savefig(save_path)
    return micro_auc


def plot_cost(training, validation, name, epochs, best_iteration):
    x = np.arange(start=0, stop=len(training), step=1).tolist()
    plt.figure()
    plt.xlim(min(x), max(x))
    plt.ylim(0, max(max(training), max(validation)) + 0.2)
    plt.plot(x, training, color='blue', linestyle='-', label='training')
    plt.plot(x, validation, color='green', linestyle='-', label='validation')
    plt.axvline(x=best_iteration, color='red')
    title = '{}: epochs={}, best_iteration={} '.format(name, epochs, best_iteration)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(name)


def plot_line(input_func, name):
    x = np.arange(start=0, stop=len(input_func), step=1).tolist()
    plt.figure()
    plt.xlim(min(x), max(x))
    plt.plot(x, input_func, color='blue', linestyle='-', label='test')
    plt.axhline(y=np.mean(input_func), color='red')
    plt.title(name)
    plt.xlabel('Cross Validation Pairs')
    plt.legend(loc='best')
    plt.savefig(name)
