from textblob import TextBlob
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from math import floor
import pandas as pd
import numpy as np
import os


stop_words = TfidfVectorizer(stop_words='english').get_stop_words()
print('stop words = ', stop_words)


def generate_sentiments(input_file, output_file):
    comments = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)
    sentiments = []
    counter = 0
    for comment in comments.comment_text.values:
        sentiment = 0
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


def group_videos(input_file, output_file):
    print('grouping videos in file %s' % input_file)
    df = pd.DataFrame(columns=['video_id', 'title', 'channel_title', 'category_id',
                               'tags', 'views', 'likes', 'dislikes', 'comment_total'])
    videos = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)
    grouped = videos.groupby('video_id')
    row_counter = 0
    for name, group in grouped:
        print ('name = %s row = %d' % (name, row_counter))
        title = group['title'].iloc[0]
        channel_title = group['channel_title'].iloc[0]
        category_id = group['category_id'].iloc[0]
        tags_string = get_tags(group)
        views = group['views'].sum()
        likes = group['likes'].sum()
        dislikes = group['dislikes'].sum()
        comment_total = group['comment_total'].sum()
        row = [name, title, channel_title, category_id, tags_string, views, likes, dislikes, comment_total]
        df.loc[row_counter] = row
        row_counter += 1
    df.to_csv(output_file, encoding='utf8')


def get_tags(group):
    tags = set()
    for row_tag_string in group['tags']:
        row_full_tags = row_tag_string.split('|')  # Split using '|'
        for row_full_tag in row_full_tags:
            row_tag = row_full_tag.split()  # Split using white space
            for token in row_tag:
                clean_token = (''.join(c for c in token if c not in punctuation)).lower()
                if clean_token not in ['none'] and clean_token not in stop_words:
                    tags.add(clean_token)
    tag_list = sorted(tags)
    tags_string = '|'.join(tag_list)
    print('tags_string = %s' % tags_string)
    return tags_string


def load_videos(videos):
    fields = ['video_id', 'title', 'category_id', 'tags']
    df = pd.read_csv(videos, encoding='utf8', error_bad_lines=False, usecols=fields)
    return df

#generate_split_sentiments('../data/youtube/UScomments.csv', '../data/youtube/USComments-sentiments.csv', 0.8)
def generate_split_sentiments(input_file, output_file, split_ratio):
    print('generating sentiments for %s' % input_file)
    generate_sentiments(input_file, output_file)
    comments = pd.read_csv(output_file, encoding='utf8', error_bad_lines=False);
    input_file_name = os.path.splitext(input_file)[0]
    split_data(comments, split_ratio, input_file_name + '-train.csv', input_file_name + '-test.csv')


def merge_video_csv(videos, comments, output_file):
    print('merging %s and %s' % (videos, comments))
    videos = load_videos(videos)
    comments_data = pd.read_csv(comments, encoding='utf8', error_bad_lines=False)
    full_data = pd.merge(comments_data, videos, how='inner', on='video_id')
    full_data.to_csv(output_file, encoding='utf8')


def split_clean_videos(input_file, ratio):
    df = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)
    df = shuffle(df)
    input_file_name = os.path.splitext(input_file)[0]
    index = int(floor(df.shape[0] * ratio))
    train_videos = df.iloc[0:index]
    train_videos.to_csv(input_file_name + '_train.csv', encoding='utf8')
    test_videos = df.iloc[index:]
    test_videos.to_csv(input_file_name + '_test.csv', encoding='utf8')


def main():
    # Split GBVideos_clean into train and test
    split_clean_videos('../data/youtube/GBvideos_clean.csv', 0.5)
    import sys
    sys.exit(0)
    # Generate sentiments for US comments
    generate_split_sentiments('../data/youtube/UScomments.csv', '../data/youtube/USComments-sentiments.csv', 0.8)
    # Generate sentiments for GB comments.
    generate_split_sentiments('../data/youtube/GBcomments.csv', '../data/youtube/GBComments-sentiments.csv', 0.8)

    # Group videos by id
    group_videos('../data/youtube/USvideos.csv', '../data/youtube/USvideos_clean.csv')
    group_videos('../data/youtube/GBvideos.csv', '../data/youtube/GBvideos_clean.csv')

    # # Merge comments data and video data for train data
    # merge_video_csv('../data/youtube/USvideos_clean.csv', '../data/youtube/USComments-train.csv',
    #                 '../data/youtube/USComments-train-full.csv')
    # # Merge comments data and video data for dev data
    # merge_video_csv('../data/youtube/USvideos_clean.csv', '../data/youtube/USComments-test.csv',
    #                 '../data/youtube/USComments-test-full.csv')
    # # Merge GB comments data and video data for test data
    # merge_video_csv('../data/youtube/GBvideos_clean.csv', '../data/youtube/GBComments-test.csv',
    #                 '../data/youtube/GBComments-test-full.csv')


if __name__ == '__main__':
    main()
