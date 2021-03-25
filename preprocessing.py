import re
import argparse
import pandas as pd
import pickle as pkl
import numpy as np
from datetime import datetime as dt
from emoji_util import load_emojis

URL_REGEX = re.compile(r'\w+:\/\/\S+')
USER_HANDLE_REGEX = re.compile(r'@[\w\d]+')
HASHTAG_REGEX = re.compile(r'#[\w\d]+')

def get_raw_tweets(file='tweets.txt'):
    '''
    function for simply loading raw tweets file
    and splitting tweets at delimiter
    '''
    text_file = open(file)
    raw_tweets = text_file.read()
    text_file.close()
    raw_tweets = raw_tweets.split(' <end_of_tweet>\n')
    return raw_tweets

def keep_tweet(tweet, emoji_regex=None, single_emoji=False, multiple_emoji=False):
    '''
    current criteria for dropping or keeping a tweet
    can be amended but for now: we exclude hashtags as they are OOV,
    exclude tweets referencing URLs as the content of the tweets
    are most likely tied to the links, and drop tweets less than 3
    words.
    '''
    keep = True
    if URL_REGEX.search(tweet) or HASHTAG_REGEX.search(tweet):
        keep = False
    if len(tweet.split(' ')) < 3:
        keep = False
    if single_emoji and len(emoji_regex.findall(tweet)) > 1:
        keep = False
    if multiple_emoji and len(emoji_regex.findall(tweet)) < 2:
        keep = False
    return keep

def create_emoji_sentence(tweet, emoji_regex, allow_repeats=True):
    '''
    extract emojis used in a sentence to create associated "emoji sentence"
    '''
    emojis = emoji_regex.findall(tweet)
    if not allow_repeats:
        emojis = list(set(emojis))
    emoji_sentence = ''.join(emojis)
    return emoji_sentence

def clean_tweet(tweet, emoji_regex):
    '''
    cosmetic adjustments to tweet. replace user-handles with [USER] token
    we may want to add special token to embedding representations, 
    remove emojis from original tweet (only to appear in emoji sentence counterpart),
    remove newline characters.
    '''
    cleaned_tweet = emoji_regex.sub('', tweet)
    cleaned_tweet = re.sub(r'\n', '', cleaned_tweet)
    cleaned_tweet = USER_HANDLE_REGEX.sub('[USER]', cleaned_tweet)
    return cleaned_tweet

def create_df(raw_tweets, emoji_regex, allow_repeats=True, single_emoji=False, multiple_emoji=False):
    '''
    combine above helper functions to create a base emoji df
    '''
    print('{} | Loading tweets file'.format(dt.now()))
    tweet_df = pd.DataFrame({'tweets': raw_tweets})
    print('{} | Started with {} tweets'.format(dt.now(), tweet_df.shape[0]))

    print('{} | Dropping duplicates and unusable tweets.'.format(dt.now()))
    tweet_filter = tweet_df.tweets.apply(lambda row: keep_tweet(row, emoji_regex, single_emoji, multiple_emoji))
    filtered_df = tweet_df[tweet_filter].copy()
    filtered_df.drop_duplicates(inplace=True)
    print('{} | Ending with {} tweets'.format(dt.now(), filtered_df.shape[0]))

    print('{} | Creating emoji sentence'.format(dt.now()))
    filtered_df['emoji_sentence'] = filtered_df.tweets.apply(lambda row: create_emoji_sentence(row, 
                                                                                               emoji_regex,
                                                                                               allow_repeats))

    print('{} | Cleaning tweets'.format(dt.now()))
    filtered_df['tweets'] = filtered_df.tweets.apply(lambda row: clean_tweet(row, emoji_regex))
    print('{} | Complete.'.format(dt.now()))

    return filtered_df

def shuffle_pairs(tweet, df, index):
    '''
    helper function to shuffle tweet-emoji sentence pairs to create "incorrect" sentence pairs.
    shuffling here ensures to avoid randomly pairing a tweet with emojis that overlap
    with its original emoji sentence. it's a simple approach and not guaranteed to manufacture
    'contradiction' but hopefully will help.
    '''
    correct_pair = set(list(tweet))
    similar = True
    while similar:
        idx = np.random.choice(index, 1).item()
        incorrect_pair = df.iloc[idx].emoji_sentence
        incorrect_pair_emojis = set(list(incorrect_pair))
        if not correct_pair.intersection(incorrect_pair_emojis):
            similar = False
    return incorrect_pair

def create_shuffled_df(df):
    '''
    Copy original emoji-sentence df to create a shuffled/incorrect counterpart, meant to be merged with
    original df.
    '''
    shuffled_df = df.copy()
    df_index = len(shuffled_df.index)
    print('{} | Creating incorrect sentence pairs'.format(dt.now()))
    incorrect_pairs = shuffled_df.emoji_sentence.apply(lambda row: shuffle_pairs(row, shuffled_df, df_index))
    print('{} | Finished creating incorrect pairs.'.format(dt.now()))
    shuffled_df['emoji_sentence'] = incorrect_pairs
    shuffled_df['follows?'] = 0
    return shuffled_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arrange twitter data for emoji nsp task.')
    parser.add_argument('--allow_repeats',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to allow repeat emojis in sequences, e.g. 😂😂😂.')
    parser.add_argument('--single_emoji',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Whether to drop sequences that consist of more than one emojis, e.g. 😘🌻🤗.')
    parser.add_argument('--multiple_emoji',
                        type=lambda s: s.lower().startswith('t'),
                        default=False,
                        help='Specifically filter for sequences that consist of more than one emoji, e.g. 💜😃😬.')
    args = parser.parse_args()
    raw_tweets = get_raw_tweets()
    emoji_list, _ = load_emojis()
    emojis =  '|'.join(emoji_list) 
    # excluding asterisk emoji *️⃣ since it
    # is interpreted as wildcard and causes more trouble than it's worth
    emojis = emojis[:3569] + emojis[3574:]
    EMOJI_REGEX = re.compile(emojis)
    emojinsp_df = create_df(raw_tweets, EMOJI_REGEX, args.allow_repeats, args.single_emoji, args.multiple_emoji)
    emojinsp_df['follows?'] = 1
    split_idx = int(len(emojinsp_df.index)/2)
    to_shuffle = emojinsp_df.iloc[split_idx:]
    emojinsp_df = emojinsp_df.iloc[:split_idx] 
    shuffled_emojinsp_df = create_shuffled_df(to_shuffle)
    emoji_frames = [emojinsp_df, shuffled_emojinsp_df]
    complete_emoji_df = pd.concat(emoji_frames)
    # included twice because dropna doesn't exclude all NaN values on single try
    # for whatever reason
    complete_emoji_df.dropna(inplace=True)
    if not args.allow_repeats:
        filename = 'emoji_nsp_dataset_no_repeats.csv'
    elif args.single_emoji:
        filename = 'emoji_nsp_dataset_single_emoji.csv'
    elif args.multiple_emoji:
        filename = 'emoji_nsp_dataset_multi_emoji.csv'
    else:
        filename = 'emoji_nsp_dataset.csv'
    complete_emoji_df.to_csv(filename, index = False)
    print('{} | Saved {}, containing {} rows'.format(dt.now(), filename, complete_emoji_df.shape[0]))


