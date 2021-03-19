import time
import argparse
import numpy as np
from credentials import * 
from tweepy import Stream, API, OAuthHandler
from tweepy.streaming import StreamListener
from datetime import datetime as dt
from emoji_util import load_emojis, chunks

class TwitterListener(StreamListener):
    def __init__(self, interval_size):
        self.tweet_count = 0
        self.interval_size = interval_size
        super(TwitterListener, self).__init__()
    def on_error(self, status_code):
        print('{} | status: {}'.format(dt.now(), status_code))
        return False
    def on_status(self, status):
        self.tweet_count += 1
        if self.tweet_count < self.interval_size:
            if not status.truncated:
                tweet = status.text
            else:
                tweet = status.extended_tweet['full_text']
            print(tweet)
            with open('tweets.txt', 'a') as file:
                file.write(tweet+' <end_of_tweet>\n')
            return True
        else:
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collect tweets using tweepy.')
    parser.add_argument('--interval_size', 
                        type=int, 
                        help='Amount of tweets to collect for each emoji chunk.',
                        default=500)
    parser.add_argument('--sleep_time',
                        type=int,
                        help='Amount of time (in seconds) to sleep between each chunk.\
                              Done to avoid rate limiting, i.e. response status 420',
                        default=30)
    args = parser.parse_args()

    print('{} | Loading emoji pickle'.format(dt.now()))
    emojis, _ = load_emojis()
    emojis = np.asarray(emojis)
    np.random.shuffle(emojis)
    emojis = emojis.tolist()
    # 400 emojis at a time is the threshold before receiving
    # a 413-status error ('request too large')
    emoji_chunks = list(chunks(emojis, 400))
    try:
        for chunk in emoji_chunks:
            auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
            auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
            api = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
            # stream = Stream(auth=api.auth, listener=Listener())
            stream = Stream(auth=api.auth, listener=TwitterListener(interval_size=args.interval_size))
            print('{} | Started streaming...'.format(dt.now()))
            stream.filter(languages=['en'], track=chunk)
            stream.disconnect()
            print('{} | Proceeding to next chunk. Sleeping for {} seconds first...'.format(dt.now(), args.sleep_time))
            time.sleep(args.sleep_time)
        print('{} | Finished streaming.'.format(dt.now()))
    except KeyboardInterrupt:
        print('{} | Stopped streaming.'.format(dt.now()))


