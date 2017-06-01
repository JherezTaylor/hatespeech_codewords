# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Define project settings
"""
import os

DIR = os.path.dirname
# Jump up three directories
PATH = os.path.join(DIR(DIR(DIR(__file__))), os.path.join("data"))

JSON_PATH = PATH + "/hatebase_data/json/"
CSV_PATH = PATH + "/hatebase_data/csv/"
OUTPUT_PATH = PATH + "/output/"
TWITTER_SEARCH_PATH = PATH + "/search_streaming/"
DATASET_PATH = PATH + "/datasets/"
WORDLIST_PATH = PATH + "/wordlists/"
MODEL_PATH = PATH + "/persistence/"
EMBEDDING_INPUT = PATH + "/neural_embedding_data/"
EMBEDDING_MODELS = PATH + "/persistence/word_embeddings"
CONLL_PATH = PATH + "/conll_data/"
PLOTLY_PATH = PATH + "/plotly_output/"
DB_URL = os.environ[
    "MONGODB_URL"] if "MONGODB_URL" in os.environ else "NOT SET"
ES_URL = os.environ["ES_URL"] if "ES_URL" in os.environ else "NOT SET"
MONGO_USER = os.environ[
    "MONGO_USER"] if "MONGO_USER" in os.environ else "NOT SET"
MONGO_PW = os.environ["MONGO_PW"] if "MONGO_PW" in os.environ else "NOT SET"
TWITTER_APP_KEY = os.environ[
    "TWITTER_APP_KEY"] if "TWITTER_APP_KEY" in os.environ else "NOT SET"
TWITTER_APP_SECRET = os.environ[
    "TWITTER_APP_SECRET"] if "TWITTER_APP_SECRET" in os.environ else "NOT SET"
TWITTER_OAUTH = os.environ[
    "TWITTER_OAUTH"] if "TWITTER_OAUTH" in os.environ else "NOT SET"
TWITTER_OAUTH_SECRET = os.environ[
    "TWITTER_OAUTH_SECRET"] if "TWITTER_OAUTH_SECRET" in os.environ else "NOT SET"
PUSHBULLET_API_KEY = os.environ[
    "PUSHBULLET_API_KEY"] if "PUSHBULLET_API_KEY" in os.environ else "NOT SET"
MONGO_SOURCE = os.environ[
    "MONGO_SOURCE"] if "MONGO_SOURCE" in os.environ else "NOT SET"
DB_AUTH_SOURCE = "admin"
# DB_URL = os.environ["LOCAL_MONGO"]
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
