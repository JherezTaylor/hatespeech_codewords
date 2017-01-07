# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Define project settings
"""
import os

DIR = os.path.dirname
## Jump up three directories
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join("data"))

JSON_PATH = PATH + "/hatebase_data/json/"
CSV_PATH = PATH + "/hatebase_data/csv/"
DATA_PATH = PATH + "/output/"
DB_URL = os.environ["MONGODB_URL"]
PUSHBULLET_API_KEY = os.environ["PUSHBULLET_API_KEY"]
MONGO_SOURCE = os.environ["MONGO_SOURCE"]
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50
GARBAGE_TWEET_DIFF = 0.4
TWITTER_APP_KEY = os.environ["TWITTER_APP_KEY"]
TWITTER_APP_SECRET = os.environ["TWITTER_APP_SECRET"]
TWITTER_OAUTH = os.environ["TWITTER_OAUTH"]
TWITTER_OAUTH_SECRET = os.environ["TWITTER_OAUTH_SECRET"]
