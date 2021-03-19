# Emoji NSP
## Getting Started
1. Make sure to install related dependencies
`pip install -r requirements.txt`
2. You'll also need to _Create an app_ [here](https://developer.twitter.com/en/apps)--a Twitter account is required to do this, so if you haven't done so already, create an account. Once you've created your app, Click into `Details` > `Keys and tokens`. Here you'll create your consumer API keys, access token, and access secret. You'll need to add these keys to `credentials.py` in order to collect tweets. 

## Scripts explained
* `emoji_util.py`: contains some helper functions used in our main `collect_tweets.py` but the main purpose of this script is to build a collection of all emojis from http://www.unicode.org/emoji/charts/full-emoji-list.html. In turn, this is used to filter for emojis when opening a Twitter stream. I've already done this step and saved the lists as pickled files, so there's no need to run this script on its own. 
* `collect_tweets.py`: this is the main script you'll be running to collect tweets containing emojis. You can inspect what command-line arguments it takes by running `python collect_tweets.py --help`. 
    * `--interval_size`: number of tweets to collect per iteration, default to 500
    * `--sleep_time`: amount of time in seconds to sleep between each iteration, default to 30 seconds
 
NOTE: There appears to be a limit to the number of words one can filter for when opening a Twitter stream (roughly 400 words). To account for this, `collect_tweets.py` iterates through our collection of emojis in chunks of size-400. Since there are ~3600 emojis, there are 9 total iterations.