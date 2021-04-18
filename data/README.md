# Emoji Datasets
This directory contains the various datasets we used to accomplish our emoji-NSP task. However, given that Twitter has specific policies around the distribution of tweets pulled through their API, we have decided to remove the csv files containing the training, validation, and test sets. NOTE: this is to avoid any potential DMCA issues on the repo as it is made publicly available. They can be provided to the graders upon request. 

The directories are self-explanatory
* `full_data` - includes data with mixed emoji usage, i.e. at least one emoji.
* `handcrafted` - in order to evaluate the model's performance more rigorously we carefully construct examples
* `multi_emoji` - includes data with tweets using more than one emoji, i.e. at least two per tweet
* `no_repeats` - eliminates repeated emojis, e.g. '😂😂😂' becomes '😂'
* `single_emoji` - tweets that use exactly one emoji

