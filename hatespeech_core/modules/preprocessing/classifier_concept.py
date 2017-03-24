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
from pymongo import InsertOne
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


@profile
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
    _cls, _pv = load_emotion_classifier()

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    query = {}
    query["filter"] = {"text": {"$ne": None}}
    query["projection"] = {"text": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True
    cursor = mongo_base.finder(connection_params, query, False)

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    cursor_1, cursor_2 = itertools.tee(cursor)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text["text"] for tweet_text in cursor_2)

    operations = []
    staging = []
    emotion_vector = []
    count = 0

    # https://github.com/explosion/spaCy/issues/172
    docs = nlp.pipe(tweet_texts, batch_size=10000, n_threads=3)
    for object_id, doc in zip(object_ids, docs):
        emotion_vector.append(doc.text)
        # _emotion_vec = _pv.transform([doc.text])
        count += 1
        print("count ", count)
        # TODO 117 seconds
        # Construct a new tweet object to be appended
        parsed_tweet = {}
        parsed_tweet["_id"] = object_id
        parsed_tweet["text"] = doc.text
        parsed_tweet["dependencies"] = [{"text": token.lower_, "lemma": token.lemma_, "pos": token.tag_,
                                         "dependency": token.dep_, "root": token.head.lower_} for token in doc if not token.is_punct]
        parsed_tweet["noun_chunks"] = [
            {"text": np.text.lower(), "root": np.root.head.text.lower()} for np in doc.noun_chunks]
        parsed_tweet["brown_cluster_ids"] = [
            token.cluster for token in doc if token.cluster != 0]
        parsed_tweet["tokens"] = list(set([token.lower_ for token in doc if not(
            token.is_stop or token.is_punct or token.lower_ == "rt" or token.is_digit or token.prefix_ == "#")]))
        parsed_tweet["hs_keyword_matches"] = list(set(
            parsed_tweet["tokens"]).intersection(hs_keywords))
        parsed_tweet["hs_keyword_count"] = len(
            parsed_tweet["hs_keyword_matches"])
        parsed_tweet["hs_keywords"] = True if parsed_tweet[
            "hs_keyword_count"] > 0 else False
        # parsed_tweet["related_keywords"] = [[w.lower_ for w in text_preprocessing.get_similar_words(
        # nlp.vocab[token], settings.NUM_SYNONYMS)] for token in
        # text_preprocessing.get_keywords(doc)]
        parsed_tweet["unknown_words"] = [
            token.lower_ for token in doc if token.is_oov and token.prefix_ != "#"]
        parsed_tweet["unknown_words_count"] = len(
            parsed_tweet["unknown_words"])
        parsed_tweet["comment_length"] = len(doc)
        parsed_tweet["avg_token_length"] = round(
            sum(len(token) for token in doc) / len(doc), 0) if len(doc) > 0 else 0
        parsed_tweet[
            "uppercase_token_count"] = text_preprocessing.count_uppercase_tokens(doc)
        parsed_tweet["bigrams"] = text_preprocessing.create_ngrams(
            doc.text.split(), 2)
        parsed_tweet["trigrams"] = text_preprocessing.create_ngrams(
            doc.text.split(), 3)
        parsed_tweet["char_trigrams"] = text_preprocessing.create_character_ngrams(
            doc.text.split(), 3)
        parsed_tweet["char_quadgrams"] = text_preprocessing.create_character_ngrams(
            doc.text.split(), 4)
        parsed_tweet["char_pentagrams"] = text_preprocessing.create_character_ngrams(
            doc.text.split(), 5)
        parsed_tweet["hashtags"] = [
            token.text for token in doc if token.prefix_ == "#"]
        staging.append(parsed_tweet)

        if len(staging) == settings.BULK_BATCH_SIZE:
            operations = unpack_emotions(staging, emotion_vector, _pv, _cls)
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []
            staging = []
            emotion_vector = []
    if staging:
        operations = unpack_emotions(staging, emotion_vector, _pv, _cls)
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

    emotion_vector = pd.Series(
        emotion_vector, index=range(0, len(emotion_vector)))
    emotion_vector = _pv.transform(emotion_vector)
    emotion_coverage = _cls.get_top_classes(
        emotion_vector, ascending=True, n=2)
    emotion_min_score = _cls.get_max_score_class(emotion_vector)

    operations = []
    for idx, parsed_tweet in enumerate(staging):
        parsed_tweet["emotions"] = {}
        parsed_tweet["emotions"]["first"] = emotion_coverage[idx][0]
        parsed_tweet["emotions"]["second"] = emotion_coverage[idx][1]
        parsed_tweet["emotions"]["min"] = emotion_min_score[idx]
        operations.append(InsertOne(parsed_tweet))
    return operations


def start_job():
    """Run operations"""
    connection_params = ["twitter", "test_suite", "feature_test"]
    nlp = init_nlp_pipeline()
    tweet = {}
    tweet["text"] = "I'm here :) :D 99"
    tweet["id"] = 7849
    # tweet_list = ["I'm here :) :D 99", "get rekt",
    #               "lol hi", "just a prank bro", "#squadgoals okay"]
    # extract_lexical_features_test(nlp, tweet_list)
    feature_extraction_pipeline(connection_params, nlp)
