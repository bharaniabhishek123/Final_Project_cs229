from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from timeit import default_timer as timer
from sklearn import decomposition
import numpy as np
import pandas as pd
import sys


def read_data(data, y_column):
    data_y = []
    data_tags = []
    for index, row in data.iterrows():
        #print(row)
        tags = row.tags
        if tags is not np.nan:
            data_y.append(row[y_column])
            data_tags.append(tags)
    return data_tags, data_y


def fit_clf(X, Y, classifier, name, tfidf_flag=True, dimensions=None):
    print('Fitting %s' % name)
    start = timer()
    tfidf = TfidfTransformer()
    if dimensions is None and not tfidf_flag:
        text_clf = Pipeline([('vect', CountVectorizer()), ('clf', classifier)])
    elif dimensions is not None and not tfidf_flag:
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        text_clf = Pipeline([('vect', CountVectorizer()), ('svd', svd), ('clf', classifier)])
    elif dimensions is None and tfidf_flag:
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf), ('clf', classifier)])
    elif dimensions is not None and tfidf_flag:
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf), ('svd', svd), ('clf', classifier)])
    text_clf.fit(X, Y)
    end = timer()
    print('Execution time = %f' % (end - start))
    return text_clf


def predict(X, Y, clf):
    predicted = clf.predict(X)
    accuracy = np.mean(predicted == Y)
    return accuracy


def process_tags(tags_list):
    for row in range(len(tags_list)):
        tags_list[row] = tags_list[row].replace('|', ' ')


def predict_categories(train, dev, test):
    y_column = 'category_id'
    train_tags, train_y = read_data(train, y_column)
    process_tags(train_tags)
    dev_tags, dev_y = read_data(dev, y_column)
    process_tags(dev_tags)
    test_tags, test_y = read_data(test, y_column)
    process_tags(test_tags)

    dimensions = None
    tfidf = False
    clf = fit_clf(train_tags, train_y, LogisticRegression(solver='newton-cg', multi_class='multinomial', tol=0.01, max_iter=30),
                  'LogisticRegression', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('LogisticRegression Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('LogisticRegression Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('LogisticRegression Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_tags, train_y, RidgeClassifier(), 'RidgeClassifier', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('RidgeClassifier Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('RidgeClassifier Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('RidgeClassifier Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_tags, train_y, BernoulliNB(), 'BernoulliNB', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('BernoulliNB Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('BernoulliNB Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('BernoulliNB Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_tags, train_y, LinearSVC(random_state=0), 'LinearSVC', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('LinearSVC Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('LinearSVC Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('LinearSVC Test Accuracy = ', test_accuracy)


def main():
    fields = ['video_id', 'category_id', 'tags']
    train = pd.read_csv('../data/youtube/USvideos_clean.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    print('train csv read')
    dev = pd.read_csv('../data/youtube/GBvideos_clean_train.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    print('dev csv read')
    test = pd.read_csv('../data/youtube/GBvideos_clean_test.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    print('dev csv read')
    predict_categories(train, dev, test)


if __name__ == '__main__':
    main()

