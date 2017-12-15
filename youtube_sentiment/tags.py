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
import matplotlib.pyplot as plt
import random
import sys


def read_data(data, y_column, y_filter=None):
    data_y = []
    data_tags = []
    for index, row in data.iterrows():
        if y_filter is None or row[y_column] in y_filter:
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


def predict_random(X, Y, clf):
    predicted = clf.predict(X)
    unique_y = list(set(Y))
    rand_Y = [unique_y[random.randint(0, len(unique_y) - 1)] for i in range(len(Y))]
    accuracy = np.mean(predicted == rand_Y)
    return accuracy


def predict(X, Y, clf):
    predicted = clf.predict(X)
    accuracy = np.mean(predicted == Y)
    return accuracy


def process_tags(tags_list):
    for row in range(len(tags_list)):
        tags_list[row] = tags_list[row].replace('|', ' ')


def predict_categories(train, dev, test, cat_filter=None):
    res = {}
    print('Number of categories used = %d' % cat_filter.size)
    y_column = 'category_id'
    train_tags, train_y = read_data(train, y_column, cat_filter)
    process_tags(train_tags)
    dev_tags, dev_y = read_data(dev, y_column, cat_filter)
    process_tags(dev_tags)
    test_tags, test_y = read_data(test, y_column, cat_filter)
    process_tags(test_tags)

    dimensions = None
    tfidf = False
    clf = fit_clf(train_tags, train_y, LogisticRegression(solver='newton-cg', C=1.0, multi_class='multinomial', tol=0.01, max_iter=30),
                  'LogisticRegression', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('LogisticRegression Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('LogisticRegression Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('LogisticRegression Test Accuracy = ', test_accuracy)
    res['LogisticRegression'] = [train_accuracy, dev_accuracy, test_accuracy]

    clf = fit_clf(train_tags, train_y, RidgeClassifier(alpha=10), 'RidgeClassifier', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('RidgeClassifier Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('RidgeClassifier Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('RidgeClassifier Test Accuracy = ', test_accuracy)
    res['RidgeClassifier'] = [train_accuracy, dev_accuracy, test_accuracy]

    clf = fit_clf(train_tags, train_y, BernoulliNB(), 'BernoulliNB', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('BernoulliNB Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('BernoulliNB Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('BernoulliNB Test Accuracy = ', test_accuracy)
    res['BernoulliNB'] = [train_accuracy, dev_accuracy, test_accuracy]

    clf = fit_clf(train_tags, train_y, LinearSVC(random_state=0, C=.05), 'LinearSVC', tfidf_flag=tfidf, dimensions=dimensions)
    train_accuracy = predict(train_tags, train_y, clf)
    print ('LinearSVC Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_tags, dev_y, clf)
    print ('LinearSVC Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_tags, test_y, clf)
    print ('LinearSVC Test Accuracy = ', test_accuracy)
    res['LinearSVC'] = [train_accuracy, dev_accuracy, test_accuracy]
    return res


def get_categories(train, num_of_categories=None):
    group_size = train.groupby('category_id').size().sort_values(ascending=False)
    cat_num = group_size.size
    print('Number of categories = %d' % cat_num)
    num_of_categories = cat_num if num_of_categories is None else num_of_categories
    num_of_categories = min(cat_num, num_of_categories)
    categories = group_size.keys()[0:num_of_categories]
    return cat_num, categories


def plot(X, Y_dict, title, file_name):
    colors = ['green', 'blue', 'red', 'brown']
    plt.figure()
    counter = 0
    for classifier in Y_dict:
        plt.plot(X, Y_dict[classifier], color=colors[counter], label=classifier)
        counter += 1
    plt.xlabel('Number of categories')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.savefig(file_name)
    pass


def main():
    fields = ['video_id', 'category_id', 'tags']

    # train = pd.read_csv('../data/youtube/USvideos_clean.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    # print('train csv read')
    # dev = pd.read_csv('../data/youtube/GBvideos_clean_train.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    # print('dev csv read')
    # test = pd.read_csv('../data/youtube/GBvideos_clean_test.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    # print('dev csv read')

    train = pd.read_csv('../data/youtube/USvideos_clean_train.csv', encoding='utf8', error_bad_lines=False, usecols=fields)
    print('train csv read')
    dev = pd.read_csv('../data/youtube/USvideos_clean_dev.csv', encoding='utf8', error_bad_lines=False,
                      usecols=fields)
    print('dev csv read')
    test = pd.read_csv('../data/youtube/USvideos_clean_test.csv', encoding='utf8', error_bad_lines=False,
                       usecols=fields)
    print('dev csv read')

    train_acc_dict = {}
    test_acc_dict = {}
    cat_num_list = []
    cat_num, _ = get_categories(train)
    for i in range(cat_num, cat_num + 1):
    #for i in range(2, 3):
        cat_num_list.append(i)
        _, cat_filter = get_categories(train, i)
        acc_res = predict_categories(train, dev, test, cat_filter=cat_filter)
        for classifier in acc_res:
            train_acc = acc_res[classifier][0]  # get train accuracy for this classifier
            train_acc_list = train_acc_dict.get(classifier, [])
            train_acc_list.append(train_acc)
            train_acc_dict[classifier] = train_acc_list

            test_acc = acc_res[classifier][2]  # get test accuracy for this classifier
            test_acc_list = test_acc_dict.get(classifier, [])
            test_acc_list.append(test_acc)
            test_acc_dict[classifier] = test_acc_list

    #plot(cat_num_list, train_acc_dict, 'Train Accuracy vs. Number of Categories', 'cat-num-train-acc.png')
    #plot(cat_num_list, test_acc_dict, 'Test Accuracy vs. Number of Categories', 'cat-num-test-acc.png')



if __name__ == '__main__':
    main()

