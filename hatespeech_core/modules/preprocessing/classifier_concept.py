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

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    cursor_1, cursor_2 = itertools.tee(cursor)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text["text"] for tweet_text in cursor_2)

    docs = nlp.pipe(tweet_texts, batch_size=10000, n_threads=3)
    for object_id, doc in zip(object_ids, docs):
        print(object_id, doc.text)
        # _emotion_vec = _pv.transform([doc.text])
        # doc["emotion_coverage"] = _cls.get_top_classes(_emotion_vec, ascending=True, n=2)
        # doc["min_emotion"] = _cls.get_max_score_class(_emotion_vec)


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
    