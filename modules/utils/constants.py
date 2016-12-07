# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Define project constants
"""
import os

DIR = os.path.dirname
## Jump up three directories
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join("Data"))

JSON_PATH = PATH + "/HatebaseData/json/"
CSV_PATH = PATH + "/HatebaseData/csv/"
DATA_PATH = PATH + "/Output/"
DB_URL = os.environ["MONGODB_URL"]
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50
