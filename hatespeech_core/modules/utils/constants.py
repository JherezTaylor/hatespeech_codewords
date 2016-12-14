# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Define project constants
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
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50