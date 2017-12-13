from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import decomposition
from timeit import default_timer as timer
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def read_data(data, y_column):
    data_y = []
    data_comments = []
    for index, row in data.iterrows():
        comment = row.comment_text
        if comment is not np.nan:
            data_y.append(row[y_column])
            data_comments.append(comment)
    return data_comments, data_y


def fit_clf(X, Y, classifier, dimensions=None):
    print('fitting %s, ' % classifier)
    start = timer()
    tfidf = TfidfTransformer()
    svd = decomposition.TruncatedSVD(n_components=dimensions)
    if dimensions is None:
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf),
                             ('clf', classifier)])
    else:
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf),
                             ('svd', svd), ('clf', classifier)])
    text_clf.fit(X, Y)
    end = timer()
    print('Execution time = ', end - start)
    return text_clf


def predict(X, Y, clf):
    predicted = clf.predict(X)
    accuracy = np.mean(predicted == Y)
    return accuracy


def add_to_plot(X_test, y_test, name, clf, ax1):
    if hasattr(clf, "predict_proba"):
        prob_pos = clf.predict_proba(X_test)[:, 1]
    else:
        prob_pos = clf.decision_function(X_test)
        prob_pos = \
            (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)

    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name,))


def predict_sentiments2(train, dev, test, dimensions=None, calibration=False):
    if calibration:
        plt.figure()
        ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    y_column = 'binary_sentiment'
    train_comments, train_y = read_data(train, y_column)
    vectorizer = CountVectorizer()
    count_vector = vectorizer.fit_transform(train_comments)
    transformer = TfidfTransformer()
    train_vector = transformer.fit_transform(count_vector)

    dev_comments, dev_y = read_data(dev, y_column)
    dev_count_vector = vectorizer.transform(dev_comments)
    dev_vector = transformer.transform(dev_count_vector)
    test_comments, test_y = read_data(test, y_column)
    test_count_vector = vectorizer.transform(test_comments)
    test_vector = transformer.transform(test_count_vector)
    if dimensions is not None:
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        train_vector = svd.fit_transform(train_vector)
        test_vector = svd.fit_transform(test_vector)
        dev_vector = svd.fit_transform(dev_vector)

    for name, clf in [('LogisticRegression', LogisticRegression(solver='newton-cg', multi_class='multinomial')),
                      ('RidgeClassifier', RidgeClassifier()),
                      ('BernoulliNB', BernoulliNB()),
                      ('LinearSVC', LinearSVC(random_state=0))]:
        print('Fitting %s' % name)
        start = timer()
        clf.fit(train_vector, train_y)
        print('Training time of %s = %f' % (name, timer() - start))
        if calibration:
            add_to_plot(dev_vector, dev_y, name, clf, ax1)
        for vector, y, data_type in [(train_vector, train_y, 'Train'),
                                     (dev_vector, dev_y, 'Dev'),
                                     (test_vector, test_y, 'Test')]:
            predicted = clf.predict(vector)
            accuracy = np.mean(predicted == y)
            print('%s %s Accuracy = %f' % (name, data_type, accuracy))

    if calibration:
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        plt.tight_layout()
        plt.savefig('calibration.png')
        plt.show()



def predict_sentiments(train, dev, test):
    y_column = 'binary_sentiment'
    train_comments, train_y = read_data(train, y_column)
    dev_comments, dev_y = read_data(dev, y_column)
    test_comments, test_y = read_data(test, y_column)


    dimensions = None
    clf = fit_clf(train_comments, train_y, LogisticRegression(solver='newton-cg', multi_class='multinomial'), dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LogisticRegression Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('LogisticRegression Dev Accuracy = ', dev_accuracy)

    test_accuracy = predict(test_comments, test_y, clf)
    print ('LogisticRegression Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, RidgeClassifier(), dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('RidgeClassifier Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('RidgeClassifier Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('RidgeClassifier Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, BernoulliNB(), dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('BernoulliNB Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('BernoulliNB Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('BernoulliNB Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, LinearSVC(random_state=0), dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LinearSVC Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('LinearSVC Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('LinearSVC Test Accuracy = ', test_accuracy)


def main():
    fields = ['comment_text', 'binary_sentiment']
    #train = pd.read_csv('../data/youtube/USComments-train-full.csv', encoding='utf-8', error_bad_lines=False, usecols=fields)
    train = pd.read_csv('../data/youtube/USComments-train.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    dev = pd.read_csv('../data/youtube/USComments-test.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    test = pd.read_csv('../data/youtube/GBComments-test.csv', encoding='utf8', error_bad_lines=False, usecols=fields)

    #predict_sentiments(train, dev, test)
    calibration = False
    dimensions = None
    if calibration:
        train = train[train.binary_sentiment != 0]
        dev = dev[dev.binary_sentiment != 0]
        test = test[test.binary_sentiment != 0]
    predict_sentiments2(train, dev, test, dimensions=dimensions, calibration=calibration)


if __name__ == '__main__':
    main()

