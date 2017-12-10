from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.calibration import calibration_curve
from timeit import default_timer as timer
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def load_videos(videos):
    fields = ['video_id', 'title', 'channel_title', 'category_id', 'tags', 'views', 'likes', 'dislikes', 'comment_total']
    df = pd.read_csv(videos, encoding='utf8', error_bad_lines=False, usecols=fields)
    print('df shape, ', df.shape)
    return df


def read_data(data, y_column):
    data_y = []
    data_comments = []
    for index, row in data.iterrows():
        #print(row)
        comment = row.comment_text
        if comment is not np.nan:
            data_y.append(row[y_column])
            data_comments.append(comment)
    return data_comments, data_y


def fit_clf(X, Y, classifier, dimensions=None):
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


def predict_categories(train, test):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    y_column = 'category_id'
    train_comments, train_y = read_data(train, y_column)
    test_comments, test_y = read_data(test, y_column)

    dimensions = None
    clf = fit_clf(train_comments, train_y, LogisticRegression(solver='newton-cg', multi_class='multinomial', tol=0.01, max_iter=20), dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LogisticRegression Train Accuracy = ', train_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('LogisticRegression Dev Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, RidgeClassifier(), dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('RidgeClassifier Train Accuracy = ', train_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('RidgeClassifier Dev Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, BernoulliNB(), dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('BernoulliNB Train Accuracy = ', train_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('BernoulliNB Dev Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, LinearSVC(random_state=0), dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LinearSVC Train Accuracy = ', train_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('LinearSVC Dev Accuracy = ', test_accuracy)


def main():
    train = pd.read_csv('../data/youtube/USComments-train-full.csv', encoding='utf8', error_bad_lines=False)
    test = pd.read_csv('../data/youtube/USComments-test-full.csv', encoding='utf8', error_bad_lines=False)
    predict_categories(train, test)


if __name__ == '__main__':
    main()

