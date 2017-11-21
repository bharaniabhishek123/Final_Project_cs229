from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_sentiments(input_file, output_file):
    comments = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False);

    sentiments = []
    counter = 0
    for comment in comments.comment_text.values:
        sentiment = 0;
        try:
            sentiment = TextBlob(comment).sentiment.polarity
        except:
            pass
        sentiments.append(sentiment)
        counter += 1
        if counter % 50000 == 0:
            print '%d comments processed so far.' % counter


    print '%d comments processed in total.' % counter
    comments['sentiment'] = sentiments
    binarySentiments = []
    for sentiment in sentiments:
        binary = 0
        if sentiment > 0:
            binary = 1
        elif sentiment < 0:
            binary = -1
        binarySentiments.append(binary)

    comments['binary_sentiment'] = binarySentiments
    comments.to_csv(output_file, encoding='utf8')

def split_data(comments, split, train_file, test_file):
    neutrals = comments[comments.binary_sentiment == 0]
    mask = np.random.rand(len(neutrals)) < split
    neutrals_train = neutrals[mask]
    neutrals_test = neutrals[~mask]
    neutrals_train.to_csv(train_file, encoding='utf8')
    neutrals_test.to_csv(test_file, encoding='utf8')
    with open(train_file, 'a') as trainf:
        with open(test_file, 'a') as testf:
            positives = comments[comments.binary_sentiment == 1]
            mask = np.random.rand(len(positives)) < split
            positives_train = positives[mask]
            positives_test = positives[~mask]
            positives_train.to_csv(trainf, encoding='utf8', header=False)
            positives_test.to_csv(testf, encoding='utf8', header=False)
            negatives = comments[comments.binary_sentiment == -1]
            mask = np.random.rand(len(negatives)) < split
            negatives_train = negatives[mask]
            negatives_test = negatives[~mask]
            negatives_train.to_csv(trainf, encoding='utf8', header=False)
            negatives_test.to_csv(testf, encoding='utf8', header=False)

def read_data(data):
    data_y = []
    data_comments = []
    for index, row in data.iterrows():
        comment = row.comment_text
        if comment is not np.nan:
            data_y.append(row['binary_sentiment'])
            data_comments.append(comment)
    return data_comments, data_y

def main():
    #generate_sentiments('../data/youtube/UScomments.csv', '../data/youtube/USComments-sentiments.csv')
    #us_comments = pd.read_csv('../data/youtube/USComments-sentiments.csv', encoding='utf8', error_bad_lines=False);
    #split_data(us_comments, 0.8, '../data/youtube/USComments-train.csv', '../data/youtube/USComments-test.csv')

    #generate_sentiments('../data/youtube/GBcomments.csv', '../data/youtube/GBComments-sentiments.csv')
    #gb_comments = pd.read_csv('../data/youtube/GBComments-sentiments.csv', encoding='utf8', error_bad_lines=False);
    #split_data(gb_comments, 0.8, '../data/youtube/GBComments-train.csv', '../data/youtube/GBComments-test.csv')

    train = pd.read_csv('../data/youtube/USComments-train.csv', encoding='utf8', error_bad_lines=False);
    train_comments, train_y = read_data(train)
    test = pd.read_csv('../data/youtube/USComments-test.csv', encoding='utf8', error_bad_lines=False);
    test_comments, test_y = read_data(test)
    dev = pd.read_csv('../data/youtube/GBComments-test.csv', encoding='utf8', error_bad_lines=False);
    dev_comments, dev_y = read_data(dev)

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RidgeClassifier())])
    text_clf.fit(train_comments, train_y)
    predicted = text_clf.predict(test_comments)
    test_accuracy = np.mean(predicted == test_y)
    print ('RidgeClassifier Test Accuracy = ', test_accuracy)
    predicted = text_clf.predict(dev_comments)
    dev_accuracy = np.mean(predicted == dev_y)
    print ('RidgeClassifier Dev Accuracy = ', dev_accuracy)

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', BernoulliNB())])
    text_clf.fit(train_comments, train_y)
    predicted = text_clf.predict(test_comments)
    test_accuracy = np.mean(predicted == test_y)
    print ('BernoulliNB Test Accuracy = ', test_accuracy)
    predicted = text_clf.predict(dev_comments)
    dev_accuracy = np.mean(predicted == dev_y)
    print ('BernoulliNB Dev Accuracy = ', dev_accuracy)

    text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LinearSVC(random_state=0))])
    text_clf.fit(train_comments, train_y)
    predicted = text_clf.predict(test_comments)
    test_accuracy = np.mean(predicted == test_y)
    print ('LinearSVC Test Accuracy = ', test_accuracy)
    predicted = text_clf.predict(dev_comments)
    dev_accuracy = np.mean(predicted == dev_y)
    print ('LinearSVC Dev Accuracy = ', dev_accuracy)

    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', DecisionTreeClassifier(random_state=0))])
    # text_clf.fit(train_comments, train_y)
    # predicted = text_clf.predict(test_comments)
    # test_accuracy = np.mean(predicted == test_y)
    # print ('DecisionTreeClassifier Test Accuracy = ', test_accuracy)
    #
    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier(max_depth=2, random_state=0))])
    # text_clf.fit(train_comments, train_y)
    # predicted = text_clf.predict(test_comments)
    # test_accuracy = np.mean(predicted == test_y)
    # print ('RandomForestClassifier Test Accuracy = ', test_accuracy)


if __name__ == '__main__':
    main()

