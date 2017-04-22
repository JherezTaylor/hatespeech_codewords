# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module serves as a proof of concept for the classifier model
"""

import itertools
import spacy
import joblib
import pandas as pd
from pymongo import InsertOne, UpdateOne
from ..utils import settings
from ..utils import file_ops
from ..utils import notifiers
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


# @notifiers.do_cprofile
def feature_extraction_pipeline(connection_params, nlp):
    """Handles the extraction of features needed for the model.
    Inserts parsed documents to database.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
            2: target_collection (str): Name of output collection.
        nlp  (spaCy model): Language processing pipeline
        tweet_generator_obj: (generator): MongoDB cursor or file loaded documents.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    target_collection = connection_params[2]
    connection_params.insert(0, client)
    # _cls, _pv = load_emotion_classifier()

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    query = {}
    query["filter"] = {"tweet_text": {"$ne": None}}
    query["projection"] = {"tweet_text": 1,
                           "does_this_tweet_contain_hate_speech": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True
    cursor = mongo_base.finder(connection_params, query, False)

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    cursor_1, cursor_2, cursor_3 = itertools.tee(cursor, 3)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text["tweet_text"] for tweet_text in cursor_2)
    contains_hs = (label["does_this_tweet_contain_hate_speech"]
                   for label in cursor_3)

    operations = []
    staging = []
    emotion_vector = []
    count = 0

    # https://github.com/explosion/spaCy/issues/172
    docs = nlp.pipe(tweet_texts, batch_size=15000, n_threads=3)
    for object_id, doc, label in zip(object_ids, docs, contains_hs):
        # emotion_vector.append(doc.text)
        count += 1
        print("count ", count)

        if str(label) == "The tweet is not offensive":
            label = "not_offensive"
        elif str(label) == "The tweet uses offensive language but not hate speech":
            label = "not_offensive"
        else:
            label = "hatespeech"

        # Construct a new tweet object to be appended
        parsed_tweet = {}
        parsed_tweet["_id"] = object_id
        parsed_tweet["text"] = doc.text
        parsed_tweet["annotation_label"] = label
        parsed_tweet["tokens"] = list([token.lower_ for token in doc if not(
            token.is_stop or token.is_punct or token.lower_ == "rt" or token.is_digit or token.prefix_ == "#")])
        parsed_tweet = text_preprocessing.prep_linguistic_features(
            parsed_tweet, hs_keywords, doc)
        parsed_tweet = text_preprocessing.prep_dependency_features(
            parsed_tweet, doc)
        # parsed_tweet["related_keywords"] = [[w.lower_ for w in text_preprocessing.get_similar_words(
        # nlp.vocab[token], settings.NUM_SYNONYMS)] for token in
        # text_preprocessing.get_keywords(doc)]

        staging.append(parsed_tweet)
        if len(staging) == 5000:
            operations = unpack_emotions(staging, emotion_vector, None, None)
            # operations = update_schema(staging)
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []
            staging = []
            emotion_vector = []
    if staging:
        operations = unpack_emotions(staging, emotion_vector, None, None)
        # operations = update_schema(staging)
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


def unpack_emotions(staging, emotion_vector, _pv, _cls):
    """Vectorize a list of tweets and return the emotion emotion_coverage
    for each entry.

    Args:
        staging (list): List of tweet objects.
        emotion_vector (list): List of tweet text.
        _pv (PatternVectorizer)
        _cls (SimpleClassifier)
    Returns:
        list of MongoDB InsertOne operations.
    """

    # emotion_vector = pd.Series(
    #     emotion_vector, index=range(0, len(emotion_vector)))
    # emotion_vector = _pv.transform(emotion_vector)
    # emotion_coverage = _cls.get_top_classes(
    #     emotion_vector, ascending=True, n=2)
    # emotion_min_score = _cls.get_max_score_class(emotion_vector)

    operations = []
    for idx, parsed_tweet in enumerate(staging):
        # parsed_tweet["emotions"] = {}
        # parsed_tweet["emotions"]["first"] = emotion_coverage[idx][0]
        # parsed_tweet["emotions"]["second"] = emotion_coverage[idx][1]
        # parsed_tweet["emotions"]["min"] = emotion_min_score[idx]
        operations.append(InsertOne(parsed_tweet))
    return operations


def update_schema(staging):
    """ Short function for appending features"""
    operations = []
    for idx, parsed_tweet in enumerate(staging):
        operations.append(UpdateOne({"_id": parsed_tweet["_id"]}, {
            "$set": {"word_dep_root": parsed_tweet["word_dep_root"], "pos_dep_rootPos": parsed_tweet["pos_dep_rootPos"], "word_root_preRoot": parsed_tweet["word_root_preRoot"], "tokens": parsed_tweet["tokens"], "conllFormat": parsed_tweet["conllFormat"], "dependency_contexts": parsed_tweet[
                "dependency_contexts"]}}, upsert=False))
    return operations


def start_feature_extraction():
    """Run operations"""
    connection_params = ["twitter", "CrowdFlower", "crowdflower_features"]
    nlp = init_nlp_pipeline()
    # tweet_list = ["I'm here :) :D 99", "get rekt",
    #               "lol hi", "just a prank bro", "#squadgoals okay"]
    # extract_lexical_features_test(nlp, tweet_list)
    feature_extraction_pipeline(connection_params, nlp)
