# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Define project constants
"""
import os

DIR = os.path.dirname
## Jump up three directories
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join('HatebaseData'))

JSON_PATH = PATH + "/json/"
CSV_PATH = PATH + "/csv/"
DATA_PATH = PATH + "/Data/"
DB_URL = ***REMOVED***
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
USER_MENTIONS_LIMIT = 50