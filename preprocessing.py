import re
import pandas as pd
import pickle as pkl
from datetime import datetime as dt
from emoji_util import load_emojis

URL_REGEX = re.compile(r'\w+:\/\/\S+')
USER_HANDLE_REGEX = re.compile(r'@[\w\d]+')
HASHTAG_REGEX = re.compile(r'#[\w\d]+')

def get_raw_tweets(file='tweets.txt'):
    text_file = open(file)
    raw_tweets = text_file.read()
    text_file.close()
    raw_tweets = raw_tweets.split(' <end_of_tweet>\n')
    return raw_tweets

def keep_tweet(tweet):
    keep = True
    if URL_REGEX.search(tweet) or HASHTAG_REGEX.search(tweet):
        keep = False
    return keep

def create_emoji_sentence(tweet, emoji_regex):
    emojis = emoji_regex.findall(tweet)
    emoji_sentence = ''.join(emojis)
    return emoji_sentence

def clean_tweet(tweet, emoji_regex):
    cleaned_tweet = emoji_regex.sub('', tweet)
    cleaned_tweet = re.sub(r'\n', '', cleaned_tweet)
    cleaned_tweet = USER_HANDLE_REGEX.sub('[USER]', cleaned_tweet)
    return cleaned_tweet

def create_df(raw_tweets, emoji_regex):
    print('{} | Loading tweets file'.format(dt.now()))
    tweet_df = pd.DataFrame({'tweets': raw_tweets})
    print('{} | Started with {} tweets'.format(dt.now(), tweet_df.shape[0]))
    print('{} | Dropping duplicates and unusable tweets.'.format(dt.now()))
    tweet_filter = tweet_df.tweets.map(keep_tweet)
    filtered_df = tweet_df[tweet_filter].copy()
    filtered_df.drop_duplicates(inplace=True)
    print('{} | Ending with {} tweets'.format(dt.now(), filtered_df.shape[0]))
    print('{} | Creating emoji sentence'.format(dt.now()))
    filtered_df['emoji_sentence'] = filtered_df.tweets.apply(lambda row: create_emoji_sentence(row, emoji_regex))
    print('{} | Cleaning tweets'.format(dt.now()))
    filtered_df['tweets'] = filtered_df.tweets.apply(lambda row: clean_tweet(row, emoji_regex))
    print('{} | Complete.'.format(dt.now()))
    filtered_df.to_csv('processed_tweets.csv', index=False)

if __name__ == '__main__':
    raw_tweets = get_raw_tweets()
    emoji_list, _ = load_emojis()
    emojis =  '|'.join(emoji_list) 
    # excluding asterisk emoji (*) since it
    # is interpreted as wildcard and causes more trouble than it's worth
    emojis = emojis[:3569] + emojis[3574:]
    EMOJI_REGEX = re.compile(emojis)
    create_df(raw_tweets, EMOJI_REGEX)
