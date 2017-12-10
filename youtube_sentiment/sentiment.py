from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn import decomposition
from timeit import default_timer as timer
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def load_videos(videos):
    fields = ['video_id', 'title', 'channel_title', 'category_id', 'tags', 'views', 'likes', 'dislikes', 'comment_total']
    df = pd.read_csv(videos, encoding='utf8', error_bad_lines=False, usecols=fields)
    print('df shape, ', df.shape)
    return df

def generate_sentiments(input_file, output_file):
    comments = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)

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
            print('%d comments processed so far.' % counter)

    print('%d comments processed in total.' % counter)
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

def read_data(data, y_column):
    data_y = []
    data_comments = []
    for index, row in data.iterrows():
        #print(row)
        comment = row.comment_text
        if comment is not np.nan:
            data_y.append(row['binary_sentiment'])
            data_comments.append(comment)
    return data_comments, data_y


def group_videos(input, output):
    df = pd.DataFrame(columns=['video_id', 'title', 'channel_title', 'category_id', 'tags', 'views', 'likes', 'dislikes', 'comment_total'])
    videos = pd.read_csv(input, encoding='utf8', error_bad_lines=False)
    grouped = videos.groupby('video_id')
    row_counter = 0
    for name, group in grouped:
        print 'name, ', name, ' row = ', row_counter
        title = group['title'].iloc[0]
        channel_title = group['channel_title'].iloc[0]
        category_id = group['category_id'].iloc[0]
        views = group['views'].sum()
        likes = group['likes'].sum()
        dislikes = group['dislikes'].sum()
        comment_total = group['comment_total'].sum()
        row = [name, title, channel_title, category_id, 'test_tag', views, likes, dislikes, comment_total]
        df.loc[row_counter] = row
        row_counter += 1
    df.to_csv(output, encoding='utf8')


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
    else:  # use decision function
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
    # generate_sentiments('../data/youtube/UScomments.csv', '../data/youtube/USComments-sentiments.csv')
    # us_comments = pd.read_csv('../data/youtube/USComments-sentiments.csv', encoding='utf8', error_bad_lines=False);
    # split_data(us_comments, 0.8, '../data/youtube/USComments-train.csv', '../data/youtube/USComments-test.csv')

    # generate_sentiments('../data/youtube/GBcomments.csv', '../data/youtube/GBComments-sentiments.csv')
    # gb_comments = pd.read_csv('../data/youtube/GBComments-sentiments.csv', encoding='utf8', error_bad_lines=False);
    # split_data(gb_comments, 0.8, '../data/youtube/GBComments-train.csv', '../data/youtube/GBComments-test.csv')

    #videos_input = '../data/youtube/USvideos.csv'
    #videos_output = '../data/youtube/USvideos_clean.csv'
    #group_videos(videos_input, videos_output)

    # videos_file = '../data/youtube/USvideos_clean.csv'
    # videos = load_videos(videos_file)
    # train = pd.read_csv('../data/youtube/USComments-train.csv', encoding='utf8', error_bad_lines=False)
    # train = pd.merge(train, videos, how='inner', on='video_id')
    # train.to_csv('../data/youtube/USComments-train-full.csv', encoding='utf8')
    # test = pd.read_csv('../data/youtube/USComments-test.csv', encoding='utf8', error_bad_lines=False)
    # test = pd.merge(test, videos, how='inner', on='video_id')
    # test.to_csv('../data/youtube/USComments-test-full.csv', encoding='utf8')

    train = pd.read_csv('../data/youtube/USComments-train-full.csv', encoding='utf8', error_bad_lines=False)
    dev = pd.read_csv('../data/youtube/USComments-test-full.csv', encoding='utf8', error_bad_lines=False)
    test = pd.read_csv('../data/youtube/GBComments-test.csv', encoding='utf8', error_bad_lines=False)

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

