# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Define project settings
"""
import os
import logging
import yaml

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(filename)s %(module)s.%(funcName)s %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

DIR = os.path.dirname
# Jump up three directories
DIR_PATH = os.path.join(DIR(DIR(DIR(__file__))))
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join("data"))

try:
    with open(DIR_PATH + "/config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
except IOError as ex:
    logger.error("I/O error %s: %s", ex.errno,
                 ex.strerror, exc_info=True)

DB_URL = cfg["mongodb"]["host"]
MONGO_USER = cfg["mongodb"]["user"]
MONGO_PW = cfg["mongodb"]["passwd"]
DB_AUTH_SOURCE = cfg["mongodb"]["auth_src"]
MONGO_SOURCE = cfg["mongodb"]["source"]
ES_URL = cfg["elasticsearch"]["host"]
TWITTER_APP_KEY = cfg["twitter_api"]["app_key"]
TWITTER_APP_SECRET = cfg["twitter_api"]["app_secret"]
TWITTER_OAUTH = cfg["twitter_api"]["oauth"]
TWITTER_OAUTH_SECRET = cfg["twitter_api"]["oauth_secret"]
PUSHBULLET_API_KEY = cfg["pushbullet"]["api_key"]

JSON_PATH = PATH + "/hatebase_data/json/"
CSV_PATH = PATH + "/hatebase_data/csv/"
OUTPUT_PATH = PATH + "/output/"
TWITTER_SEARCH_PATH = PATH + "/search_streaming/"
DATASET_PATH = PATH + "/datasets/"
WORDLIST_PATH = PATH + "/wordlists/"
MODEL_PATH = PATH + "/persistence/"
EMBEDDING_INPUT = PATH + "/neural_embedding_data/"
EMBEDDING_MODELS = PATH + "/persistence/word_embeddings/"
CLASSIFIER_MODELS = PATH + "/persistence/classifiers/"
CONLL_PATH = PATH + "/conll_data/"
PLOTLY_PATH = PATH + "/plotly_output/"
HASHTAGS = "entities.hashtags"
USER_MENTIONS = "entities.user_mentions"
HASHTAG_LIMIT = 50
PNGRAM_THRESHOLD = 8
BULK_BATCH_SIZE = 10000
USER_MENTIONS_LIMIT = 50
GARBAGE_TWEET_DIFF = 0.4
NUM_SYNONYMS = 5
SPACY_EN_MODEL = "en_core_web_md"
SPACY_GLOVE_MODEL = "en_vectors_glove_md"
CRWDFLR_DATA_RAW = "data/persistence/df/crowdflower_features_raw.pkl.compressed"
CRWDFLR_DATA = "data/persistence/df/crowdflower_features.pkl.compressed"
NAACL_2016_DATA = "data/persistence/df/naacl_2016.pkl.compressed"
NLP_2016_DATA = "data/persistence/df/nlp_2016.pkl.compressed"
EMO_CLF = "data/persistence/simple_classifier_model.pkl.compressed"
EMO_PV = "data/persistence/pattern_vectorizer.pkl.compressed"
