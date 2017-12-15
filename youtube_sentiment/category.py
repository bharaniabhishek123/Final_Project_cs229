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


def fit_clf(X, Y, classifier, name, dimensions=None):
    print('Fitting %s' % name)
    start = timer()
    tfidf = TfidfTransformer()
    if dimensions is None:
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf),
                             ('clf', classifier)])
    else:
        svd = decomposition.TruncatedSVD(n_components=dimensions)
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', tfidf),
                             ('svd', svd), ('clf', classifier)])
    text_clf.fit(X, Y)
    end = timer()
    print('Execution time = %f' % (end - start))
    return text_clf


def predict(X, Y, clf):
    predicted = clf.predict(X)
    accuracy = np.mean(predicted == Y)
    return accuracy


def predict_categories(train, dev, test):

    y_column = 'category_id'
    train_comments, train_y = read_data(train, y_column)
    dev_comments, dev_y = read_data(dev, y_column)
    test_comments, test_y = read_data(test, y_column)

    dimensions = None
    clf = fit_clf(train_comments, train_y, LogisticRegression(solver='newton-cg', multi_class='multinomial', tol=0.01, max_iter=30), 'LogisticRegression', dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LogisticRegression Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('LogisticRegression Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('LogisticRegression Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, RidgeClassifier(), 'RidgeClassifier', dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('RidgeClassifier Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('RidgeClassifier Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('RidgeClassifier Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, BernoulliNB(), 'BernoulliNB', dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('BernoulliNB Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('BernoulliNB Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('BernoulliNB Test Accuracy = ', test_accuracy)

    clf = fit_clf(train_comments, train_y, LinearSVC(random_state=0), 'LinearSVC', dimensions=dimensions)
    train_accuracy = predict(train_comments, train_y, clf)
    print ('LinearSVC Train Accuracy = ', train_accuracy)
    dev_accuracy = predict(dev_comments, dev_y, clf)
    print ('LinearSVC Dev Accuracy = ', dev_accuracy)
    test_accuracy = predict(test_comments, test_y, clf)
    print ('LinearSVC Test Accuracy = ', test_accuracy)


def load_videos(videos):
    fields = ['video_id', 'title', 'category_id', 'tags']
    df = pd.read_csv(videos, encoding='utf8', error_bad_lines=False, usecols=fields)
    return df


def merge_video_csv(videos, comments, output_file):
    print('merging %s and %s' % (videos, comments))
    videos = load_videos(videos)
    comments_data = pd.read_csv(comments, encoding='utf8', error_bad_lines=False)
    full_data = pd.merge(comments_data, videos, how='inner', on='video_id')
    full_data.to_csv(output_file, encoding='utf8')


def main():
    train = pd.read_csv('../data/youtube/USComments-train-full.csv', encoding='utf8', error_bad_lines=False)
    test_full = pd.read_csv('../data/youtube/USComments-test-full.csv', encoding='utf8', error_bad_lines=False)
    middle = test_full.shape[0] / 2
    dev = test_full.iloc[0:middle]
    test = test_full.iloc[middle:]
    predict_categories(train, dev, test)


if __name__ == '__main__':
    main()

