from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import decomposition
from timeit import default_timer as timer
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from itertools import cycle

def predict_sentiments3(train, dev, test):
    y_column = 'binary_sentiment'
    train_comments, train_y = read_data(train, y_column)
    dev_comments, dev_y = read_data(dev, y_column)
    test_comments, test_y = read_data(test, y_column)

    train_y_binarize = label_binarize(train_y, classes=[-1, 0, 1])
    dev_y_binarize = label_binarize(dev_y, classes=[-1, 0, 1])
    test_y_binarize = label_binarize(test_y, classes=[-1, 0, 1])

    n_classes = dev_y_binarize.shape[1]

    #LogisticRegression

    dimensions = None

    clf = fit_clf(train_comments, train_y, LogisticRegression(solver='newton-cg', multi_class='multinomial'), dimensions=dimensions)

    train_score = clf.decision_function(train_comments)
    dev_score = clf.decision_function(dev_comments)
    test_score = clf.decision_function(test_comments)

    train_precision = dict()
    train_recall = dict()
    train_avg_precision = dict()
    dev_precision = dict()
    dev_recall = dict()
    dev_avg_precision = dict()
    test_precision = dict()
    test_recall = dict()
    test_avg_precision = dict()

    for i in range(n_classes):
        train_precision[i], train_recall[i], _ = precision_recall_curve(train_y_binarize[:, i],train_score[:, i])
        train_avg_precision[i] = average_precision_score(train_y_binarize[:, i], train_score[:, i])
        dev_precision[i], dev_recall[i], _ = precision_recall_curve(dev_y_binarize[:, i],dev_score[:, i])
        dev_avg_precision[i] = average_precision_score(dev_y_binarize[:, i], dev_score[:, i])
        test_precision[i], test_recall[i], _ = precision_recall_curve(test_y_binarize[:, i],test_score[:, i])
        test_avg_precision[i] = average_precision_score(test_y_binarize[:, i], test_score[:, i])

    train_avg_precision["micro"] = average_precision_score(train_y_binarize, train_score, average="micro")
    train_precision["micro"], train_recall["micro"], _  = precision_recall_curve(train_y_binarize.ravel(), train_score.ravel())
    print ('LogisticRegression Train Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(train_avg_precision["micro"]))
    dev_avg_precision["micro"] = average_precision_score(dev_y_binarize, dev_score, average="micro")
    dev_precision["micro"], dev_recall["micro"], _  = precision_recall_curve(dev_y_binarize.ravel(), dev_score.ravel())
    print ('LogisticRegression Dev Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(dev_avg_precision["micro"]))
    test_avg_precision["micro"] = average_precision_score(test_y_binarize, test_score, average="micro")
    test_precision["micro"], test_recall["micro"], _  = precision_recall_curve(test_y_binarize.ravel(), test_score.ravel())
    print ('LogisticRegression Test Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(test_avg_precision["micro"]))

    # setup plot details dev data
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(dev_recall["micro"], dev_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(dev_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(dev_recall[i], dev_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i-1), dev_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for LogisticRegression on dev data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_LogisticRegression_dev.png')
    plt.show()

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(test_recall["micro"], test_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(test_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(test_recall[i], test_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i-1), test_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for LogisticRegression on test data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_LogisticRegression_test.png')
    plt.show()

    #RidgeClassifier

    dimensions = None

    clf = fit_clf(train_comments, train_y, RidgeClassifier(), dimensions=dimensions)

    train_score = clf.decision_function(train_comments)
    dev_score = clf.decision_function(dev_comments)
    test_score = clf.decision_function(test_comments)

    train_precision = dict()
    train_recall = dict()
    train_avg_precision = dict()
    dev_precision = dict()
    dev_recall = dict()
    dev_avg_precision = dict()
    test_precision = dict()
    test_recall = dict()
    test_avg_precision = dict()

    for i in range(n_classes):
        train_precision[i], train_recall[i], _ = precision_recall_curve(train_y_binarize[:, i],train_score[:, i])
        train_avg_precision[i] = average_precision_score(train_y_binarize[:, i], train_score[:, i])
        dev_precision[i], dev_recall[i], _ = precision_recall_curve(dev_y_binarize[:, i],dev_score[:, i])
        dev_avg_precision[i] = average_precision_score(dev_y_binarize[:, i], dev_score[:, i])
        test_precision[i], test_recall[i], _ = precision_recall_curve(test_y_binarize[:, i],test_score[:, i])
        test_avg_precision[i] = average_precision_score(test_y_binarize[:, i], test_score[:, i])

    train_avg_precision["micro"] = average_precision_score(train_y_binarize, train_score, average="micro")
    train_precision["micro"], train_recall["micro"], _  = precision_recall_curve(train_y_binarize.ravel(), train_score.ravel())
    print ('RidgeClassifier Train Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(train_avg_precision["micro"]))
    dev_avg_precision["micro"] = average_precision_score(dev_y_binarize, dev_score, average="micro")
    dev_precision["micro"], dev_recall["micro"], _  = precision_recall_curve(dev_y_binarize.ravel(), dev_score.ravel())
    print ('RidgeClassifier Dev Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(dev_avg_precision["micro"]))
    test_avg_precision["micro"] = average_precision_score(test_y_binarize, test_score, average="micro")
    test_precision["micro"], test_recall["micro"], _  = precision_recall_curve(test_y_binarize.ravel(), test_score.ravel())
    print ('RidgeClassifier Test Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(test_avg_precision["micro"]))

    # setup plot details test data
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(dev_recall["micro"], dev_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(dev_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(dev_recall[i], dev_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i-1), dev_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for RidgeClassifier on dev data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_RidgeClassifier_dev.png')
    plt.show()

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(test_recall["micro"], test_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(test_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(test_recall[i], test_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i-1), test_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for RidgeClassifier on test data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_RidgeClassifier_test.png')
    plt.show()

    # LinearSVC

    dimensions = None

    clf = fit_clf(train_comments, train_y, LinearSVC(random_state=0), dimensions=dimensions)

    train_score = clf.decision_function(train_comments)
    dev_score = clf.decision_function(dev_comments)
    test_score = clf.decision_function(test_comments)

    train_precision = dict()
    train_recall = dict()
    train_avg_precision = dict()
    dev_precision = dict()
    dev_recall = dict()
    dev_avg_precision = dict()
    test_precision = dict()
    test_recall = dict()
    test_avg_precision = dict()

    for i in range(n_classes):
        train_precision[i], train_recall[i], _ = precision_recall_curve(train_y_binarize[:, i], train_score[:, i])
        train_avg_precision[i] = average_precision_score(train_y_binarize[:, i], train_score[:, i])
        dev_precision[i], dev_recall[i], _ = precision_recall_curve(dev_y_binarize[:, i], dev_score[:, i])
        dev_avg_precision[i] = average_precision_score(dev_y_binarize[:, i], dev_score[:, i])
        test_precision[i], test_recall[i], _ = precision_recall_curve(test_y_binarize[:, i], test_score[:, i])
        test_avg_precision[i] = average_precision_score(test_y_binarize[:, i], test_score[:, i])

    train_avg_precision["micro"] = average_precision_score(train_y_binarize, train_score, average="micro")
    train_precision["micro"], train_recall["micro"], _ = precision_recall_curve(train_y_binarize.ravel(),
                                                                                train_score.ravel())
    print('LinearSVC Train Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(
        train_avg_precision["micro"]))
    dev_avg_precision["micro"] = average_precision_score(dev_y_binarize, dev_score, average="micro")
    dev_precision["micro"], dev_recall["micro"], _ = precision_recall_curve(dev_y_binarize.ravel(), dev_score.ravel())
    print('LinearSVC Dev Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(
        dev_avg_precision["micro"]))
    test_avg_precision["micro"] = average_precision_score(test_y_binarize, test_score, average="micro")
    test_precision["micro"], test_recall["micro"], _ = precision_recall_curve(test_y_binarize.ravel(),
                                                                              test_score.ravel())
    print('LinearSVC Test Average Precision Score, micro-averaged over all classes : {0:0.3f}'.format(
        test_avg_precision["micro"]))

    # setup plot details dev data
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(dev_recall["micro"], dev_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(dev_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(dev_recall[i], dev_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i - 1), dev_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for LinearSVC on dev data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_LinearSVC_dev.png')
    plt.show()

    # setup plot details test data
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(test_recall["micro"], test_precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.3f})'
                  ''.format(test_avg_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(test_recall[i], test_precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.3f})'
                      ''.format((i - 1), test_avg_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for LinearSVC on test data')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
    plt.savefig('PrecisionRecall_LinearSVC_test.png')
    plt.show()


def main():
    fields = ['comment_text', 'binary_sentiment']
    #train = pd.read_csv('../data/youtube/USComments-train-full.csv', encoding='utf-8', error_bad_lines=False, usecols=fields)
    train = pd.read_csv('C:/Users/PengSeng/Desktop/CS 229 Machine Learning/Project/data/youtube/USComments-train-full.csv', encoding='utf8', error_bad_lines=False)
    test = pd.read_csv('C:/Users/PengSeng/Desktop/CS 229 Machine Learning/Project/data/youtube/USComments-test-full.csv', encoding='utf8', error_bad_lines=False)
    dev = pd.read_csv('C:/Users/PengSeng/Desktop/CS 229 Machine Learning/Project/data/youtube/GBComments-test.csv', encoding='utf8', error_bad_lines=False)

    #predict_sentiments(train, dev, test)
    calibration = False
    precison_recall = True
    dimensions = None
    if calibration:
        train = train[train.binary_sentiment != 0]
        dev = dev[dev.binary_sentiment != 0]
        test = test[test.binary_sentiment != 0]
    predict_sentiments2(train, dev, test, dimensions=dimensions, calibration=calibration)

    if precison_recall:
        predict_sentiments3(train, dev, test)
if __name__ == '__main__':
    main()




