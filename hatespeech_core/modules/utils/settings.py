# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Define project settings
"""
import os

DIR = os.path.dirname
## Jump up three directories
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join("data"))

JSON_PATH = PATH + "/hatebase_data/json/"
CSV_PATH = PATH + "/hatebase_data/csv/"
OUTPUT_PATH = PATH + "/output/"
TWITTER_SEARCH_PATH = PATH + "/search_streaming/"
DATASET_PATH = PATH + "/datasets/"
WORDLIST_PATH = PATH + "/wordlists/"
MODEL_PATH = PATH + "/persistence/"
DB_URL = os.environ["MONGODB_URL"]
MONGO_USER = os.environ["MONGO_USER"]
MONGO_PW = os.environ["MONGO_PW"]
DB_AUTH_SOURCE = 'admin'
# DB_URL = os.environ["LOCAL_MONGO"]
PUSHBULLET_API_KEY = os.environ["PUSHBULLET_API_KEY"]
MONGO_SOURCE = os.environ["MONGO_SOURCE"]
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
PNGRAM_THRESHOLD = 8
BULK_BATCH_SIZE = 10000
USER_MENTIONS_LIMIT = 50
GARBAGE_TWEET_DIFF = 0.4
NUM_SYNONYMS = 5
TWITTER_APP_KEY = os.environ["TWITTER_APP_KEY"]
TWITTER_APP_SECRET = os.environ["TWITTER_APP_SECRET"]
TWITTER_OAUTH = os.environ["TWITTER_OAUTH"]
TWITTER_OAUTH_SECRET = os.environ["TWITTER_OAUTH_SECRET"]
