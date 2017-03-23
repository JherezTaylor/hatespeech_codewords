# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module serves as a proof of concept for the classifier model
"""

import spacy
import joblib
from ..utils import settings
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
    return [_cls, _pv]


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
    # print(result)


def feature_extraction_pipeline(connection_params, nlp, tweet_generator_obj):
    """Handles the extraction of features needed for the model.
    Inserts parsed documents to database.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        nlp  (spaCy model): Language processing pipeline
        tweet_generator_obj: (generator): MongoDB cursor or file loaded documents.
    """

    db_name = connection_params[0]
    collection = connection_params[1]

    for doc in nlp.pipe(tweet_generator_obj, batch_size=10000, n_threads=3):
        pass


def start_job():
    nlp = init_nlp_pipeline()
    tweet_list = ["I'm here :) :D 99", "get rekt",
                  "lol hi", "just a prank bro", "#squadgoals okay"]
    extract_lexical_features_test(nlp, tweet_list)
