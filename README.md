# Emoji NSP

This repo contains a research project for DS-GA 1012 (Natural Language Understanding). Our goal was to learn rich emoji representations through an entailement-relation/ NSP-style binary classification task.  Our paper is included in the repo as well. Below is our abstract:

> NLP research on emojis has mostly focused on emotional information that emojis capture or has attempted to associate large chunks of text with a single emoji. We hypothesize that a single emoji is unable to distinctly capture and represent meaning that is context-specific when used in text-based messages. We ex- periment with an NSP-style binary classifica- tion task, leveraging different embedding tech- niques, and in support of our hypothesis, we find that our models learn better representa- tions when trained on sequences of emojis rather than single emojis.

## Getting Started
1. Make sure to install related dependencies
`pip install -r requirements.txt`
2. You'll also need to _Create an app_ [here](https://developer.twitter.com/en/apps)--a Twitter account is required to do this, so if you haven't done so already, create an account. Once you've created your app, Click into `Details` > `Keys and tokens`. Here you'll create your consumer API keys, access token, and access secret. You'll need to add these keys to `credentials.py` in order to collect tweets. 

## Scripts explained
* `emoji_util.py`: contains some helper functions used in our main `collect_tweets.py` but the main purpose of this script is to build a collection of all emojis from http://www.unicode.org/emoji/charts/full-emoji-list.html. In turn, this is used to filter for emojis when opening a Twitter stream. I've already done this step and saved the lists as pickled files, so there's no need to run this script on its own. 
* `collect_tweets.py`: this is the main script you'll be running to collect tweets containing emojis. You can inspect what command-line arguments it takes by running `python collect_tweets.py --help`. 
    * `--interval_size`: number of tweets to collect per iteration, default to 500
    * `--sleep_time`: amount of time in seconds to sleep between each iteration, default to 30 seconds
 
NOTE: There appears to be a limit to the number of words one can filter for when opening a Twitter stream (roughly 400 words). To account for this, `collect_tweets.py` iterates through our collection of emojis in chunks of size-400. Since there are roughly 3600 emojis, there are 9 total iterations.

* `preprocessing.py`: cleans tweets by removing URLs, dropping tweets less than 3 words long, replacing user-handles with `[USER]` token. constructs our four different datasets: full (single + multi), no repeats, single, multi.

## Models
### RoBERTa-based model
* `bertmoji_model.py`: includes model class for emoji-NSP, as well as trainer class for either fully executing train loop or individual train and evaluation steps.
* `bertmoji_utils.py`: helper functions for loading fine-tuned bertmoji model and loading individual examples for the purposes of probing and analyzing.
* `RoBERTa_error_analysis.ipynb` : preliminary error analysis done on RoBERTa model

### Logistic Regression model
* `logistic_regression_preprocess_embeddings.ipynb`: includes preprocessing, tokenization, averaged/concatenated embeddings for text and emojis - separately done for all four datasets.
* `logistic_regression_model_prediction.ipynb`: baseline and finetuned logistic regression model results for all four datasets, using averaged/concatenated embeddings.
* `logistic_regression_error_analysis.ipynb`: preliminary error analysis done on the best result (full data with averaged embeddings)

## Results
Detailed explanation of our experimentation, analysis, and results can be found in our paper (`nul_emojinsp.pdf`). The bertmoji (i.e. RoBERTa-base model) perfromed the best, achieving as high as `0.818` accuracy. Below is a small collection of example predictions made for text-emoji pairs
![bertmoji-predictions](/assets/bertmoji-predictions.png)