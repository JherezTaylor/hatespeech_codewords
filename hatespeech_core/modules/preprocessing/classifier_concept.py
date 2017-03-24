# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module serves as a proof of concept for the classifier model
"""

import itertools
import spacy
import joblib
from ..utils import settings
from ..utils import file_ops
from ..utils import text_preprocessing
from ..db import mongo_base
from ..utils.CustomTwokenizer import CustomTwokenizer
from ..pattern_classifier import SimpleClassifier, PatternVectorizer


def init_nlp_pipeline():
    """Initialize spaCy nlp pipeline

    Returns:
        nlp: spaCy language model
    """
    nlp = spacy.load('en', create_make_doc=CustomTwokenizer)
    return nlp


def load_emotion_classifier():
    """Loads persisted classes for the emotion classifier

    Returns:
        SimpleClassifier, PatternVectorizer classes
    """

    cls_persistence = settings.MODEL_PATH + "simple_classifier_model.pkl.compressed"
    pv_persistence = settings.MODEL_PATH + "pattern_vectorizer.pkl.compressed"
    _cls = joblib.load(cls_persistence)
    _pv = joblib.load(pv_persistence)
    return (_cls, _pv)


def extract_lexical_features_test(nlp, tweet_list):
    """Provides tokenization, POS and dependency parsing
    Args:
        nlp  (spaCy model): Language processing pipeline
    """
    result = []
    texts = (tweet for tweet in tweet_list)
    for doc in nlp.pipe(texts, batch_size=10000, n_threads=3):
        print(doc)
        for parsed_doc in doc:
            result.append((parsed_doc.orth_, parsed_doc.tag_))


def feature_extraction_pipeline(connection_params, nlp):
    """Handles the extraction of features needed for the model.
    Inserts parsed documents to database.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        nlp  (spaCy model): Language processing pipeline
        tweet_generator_obj: (generator): MongoDB cursor or file loaded documents.
    """

    client = mongo_base.connect()
    connection_params.insert(0, client)
    _cls, _pv = load_emotion_classifier()

    query = {}
    query["filter"] = {"text": {"$ne": None}}
    query["projection"] = {"text": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True
    cursor = mongo_base.finder(connection_params, query, False)

    HS_KEYWORDS = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    cursor_1, cursor_2 = itertools.tee(cursor)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text["text"] for tweet_text in cursor_2)

    staging = []
    operations = []
    # Carry out the same stage/bulk operation
    # https://github.com/explosion/spaCy/issues/172
    docs = nlp.pipe(tweet_texts, batch_size=10000, n_threads=3)
    for object_id, doc in zip(object_ids, docs):
        _emotion_vec = _pv.transform([doc.text])

        # Construct a new tweet object to be appended
        parsed_tweet = {}
        parsed_tweet["_id"] = object_id
        parsed_tweet["emotion"] = _cls.get_top_classes(
            _emotion_vec, ascending=True, n=2)
        parsed_tweet["min_emotion"] = _cls.get_max_score_class(_emotion_vec)
        parsed_tweet["dependencies"] = [[{"text": token.lower_, "lemma": token.lemma_, "pos": token.tag_, "dependency": token.dep_, "root": token.head.lower_}] for token in doc if not(token.is_punct)]
        parsed_tweet["noun_chunks"] = [[{"text": np.lower_,"root": np.root.head.lower_}] for np in doc.noun_chunks]
        parsed_tweet["brown_cluster_ids"] = [token.cluster for token in doc if token.cluster != 0]
        parsed_tweet["tokens"] = [token for token in doc if not(token.is_stop or token.is_punct or token.lower_ == "rt" or token.prefix_ == "@" or token.is_digit)]
        parsed_tweet["hs_keyword_matches"] = set(parsed_tweet["tokens"]).intersection(HS_KEYWORDS)
        parsed_tweet["hs_keyword_count"] = len(parsed_tweet["hs_keyword_matches"])
        parsed_tweet["hs_keywords"] = True if parsed_tweet["hs_keyword_count"] > 0 else False
        parsed_tweet["related_keywords"] = [[w.lower_ for w in text_preprocessing.get_similar_words(nlp.vocab[token], settings.NUM_SYNONYMS)] for token in text_preprocessing.get_keywords(doc)]
        parsed_tweet["unknown_words"] = [token.lower_ for token in doc if token.is_oov and token.prefix_ != ("@" or "#")]
        parsed_tweet["unknown_words_count"] = len(parsed_tweet["unknown_words"])
        parsed_tweet["comment_length"] = len(doc)
        parsed_tweet["avg_token_length"] = round(sum(len(token) for token in doc) / len(doc), 0)
        parsed_tweet["uppercase_token_count"] = text_preprocessing.count_uppercase_tokens(doc)
        print(object_id, doc.text)


def start_job():
    connection_params = ["twitter", "test_suite"]
    nlp = init_nlp_pipeline()
    tweet = {}
    tweet["text"] = "I'm here :) :D 99"
    tweet["id"] = 7849
    # tweet_list = ["I'm here :) :D 99", "get rekt",
    #               "lol hi", "just a prank bro", "#squadgoals okay"]
    # extract_lexical_features_test(nlp, tweet_list)
    feature_extraction_pipeline(connection_params, nlp)
