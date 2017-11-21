import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# This file contains method to generate word cloud of positive comments and negative comments

def generate_positive_WordCloud(input_file):
    df = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)
    comments = ['comment_text']
    pos_text = df.query('binary_sentiment==1').to_string(columns=comments,index=False)
        # To see all the positive comments in csv file
        #to_csv(output_file1, columns=pos_comments,encoding='utf8',index=False)

    wordcloud_pos = WordCloud().generate(pos_text)
    plt.imshow(wordcloud_pos, interpolation='bilinear')
    plt.axis("off")

    plt.savefig('pos_WordCloud')

def generate_negative_WordCloud(input_file):
    df = pd.read_csv(input_file, encoding='utf8', error_bad_lines=False)
    comments = ['comment_text']

    neg_text = df.query('binary_sentiment==-1').to_string(columns=comments, index=False)

    wordcloud_neg = WordCloud().generate(neg_text)
    plt.imshow(wordcloud_neg, interpolation='bilinear')
    plt.axis("off")

    plt.savefig('neg_WordCloud')

generate_positive_WordCloud('../data/youtube/USComments-sentiments.csv')
generate_negative_WordCloud('../data/youtube/USComments-sentiments.csv')