# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module serves as a proof of concept for the classifier model
"""

import itertools
import spacy
import joblib
from pymongo import InsertOne, UpdateOne
from ..utils import settings
from ..utils import file_ops
from ..utils import notifiers
from ..utils import text_preprocessing
from ..db import mongo_base
from ..db import elasticsearch_base
from ..utils.CustomTwokenizer import CustomTwokenizer, EmbeddingTwokenizer
from ..pattern_classifier import SimpleClassifier, PatternVectorizer


def init_nlp_pipeline(parser, tokenizer=CustomTwokenizer):
    """Initialize spaCy nlp pipeline
    The params are boolean values that determine if that feature should
    be loaded with the pipeline.

    Returns:
        nlp: spaCy language model
    """
    if parser is False:
        nlp = spacy.load(settings.SPACY_EN_MODEL, create_make_doc=tokenizer,
                         parser=False)
    else:
        nlp = spacy.load(settings.SPACY_EN_MODEL,
                         create_make_doc=tokenizer)
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
        settings.logger.info(doc)
        for parsed_doc in doc:
            result.append((parsed_doc.orth_, parsed_doc.tag_))
    return result


def run_parallel_pipeline(connection_params, method, job_details):
    """ Generic function for processing a collection in parallel.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
            2: projection (str): Document field name to return.
        job_details (list): Info to insert in job complete notifcation

    """

    client = mongo_base.connect()
    query = {}

    projection = connection_params[2]
    query["filter"] = {}
    query["projection"] = {projection: 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    connection_params.insert(0, client)
    collection_size = mongo_base.finder(connection_params, query, True)
    del connection_params[0]
    client.close()

    if collection_size == 0:
        return

    num_cores = 1
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    time1 = notifiers.time()
    joblib.Parallel(n_jobs=num_cores)(joblib.delayed(method)(
        connection_params, query, partition) for partition in partitions)
    time2 = notifiers.time()
    notifiers.send_job_completion(
        [time1, time2], [job_details[0], connection_params[1] + ": " + job_details[1]])


def feature_extraction_pipeline(connection_params, query, partition, usage=None):
    """Handles the extraction of features needed for the model.
    Inserts parsed documents to database.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
            2: projection (str): Document field name to return.
            3: target_collection (str): Name of output collection.
    """

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    nlp = init_nlp_pipeline(True)
    client = mongo_base.connect()
    db_name = connection_params[0]
    projection = connection_params[2]
    target_collection = connection_params[3]
    usage = connection_params[4]
    connection_params.insert(0, client)
    # _cls, _pv = load_emotion_classifier()

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    cursor = mongo_base.finder(connection_params, query, False)

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    cursor_1, cursor_2 = itertools.tee(cursor, 2)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text[projection] for tweet_text in cursor_2)

    operations = []
    staging = []
    emotion_vector = []
    count = 0

    # https://github.com/explosion/spaCy/issues/172
    docs = nlp.pipe(tweet_texts, batch_size=15000, n_threads=4)
    for object_id, doc in zip(object_ids, docs):
        # emotion_vector.append(doc.text)
        count += 1

        # Construct a new tweet object to be appended
        parsed_tweet = {}
        parsed_tweet["_id"] = object_id
        if usage == "conll":
            parsed_tweet = text_preprocessing.prep_dependency_features(
                parsed_tweet, doc, usage)

        else:
            parsed_tweet["text"] = doc.text
            if usage == "analysis":
                parsed_tweet["tokens"] = list([token.lower_ for token in doc if not(
                    token.is_stop or token.is_punct or token.lower_ == "rt" or token.is_digit or token.prefix_ == "#")])
                parsed_tweet = text_preprocessing.prep_linguistic_features(
                    parsed_tweet, hs_keywords, doc, usage)
                parsed_tweet = text_preprocessing.prep_dependency_features(
                    parsed_tweet, doc, usage)
            elif usage == "features":
                parsed_tweet["tokens"] = list([token.lower_ for token in doc if not(
                    token.is_stop or token.is_punct or token.lower_ == "rt" or token.is_digit or token.prefix_ == "#")])
                parsed_tweet = text_preprocessing.prep_linguistic_features(
                    parsed_tweet, hs_keywords, doc, usage)
                parsed_tweet = text_preprocessing.prep_dependency_features(
                    parsed_tweet, doc, usage)

        # parsed_tweet["related_keywords"] = [[w.lower_ for w in text_preprocessing.get_similar_words(
        # nlp.vocab[token], settings.NUM_SYNONYMS)] for token in
        # text_preprocessing.get_keywords(doc)]

        staging.append(parsed_tweet)
        if len(staging) == settings.BULK_BATCH_SIZE:
            print("count ", count)
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


def store_preprocessed_text(connection_params, query, partition):
    """ Read a MongoDB collection and store the preprocessed text
    as a separate field. Preprocessing removes URLS, numbers, and
    stopwords, normalizes @usermentions. Updates the passed collection.
    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
            2: projection (str): Document field name to return.
        query (dict): Query to execute.
        partition   (tuple): Contains skip and limit values.

    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    projection = connection_params[2]
    connection_params.insert(0, client)

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    operations = []
    nlp = init_nlp_pipeline(False, tokenizer=EmbeddingTwokenizer)
    cursor = mongo_base.finder(connection_params, query, False)

    # Makes a copy of the MongoDB cursor, to the best of my
    # knowledge this does not attempt to exhaust the cursor
    count = 0
    cursor_1, cursor_2 = itertools.tee(cursor, 2)
    object_ids = (object_id["_id"] for object_id in cursor_1)
    tweet_texts = (tweet_text[projection] for tweet_text in cursor_2)

    docs = nlp.pipe(tweet_texts, batch_size=15000, n_threads=4)
    for object_id, doc in zip(object_ids, docs):
        count += 1
        parsed_tweet = {}
        parsed_tweet["_id"] = object_id
        parsed_tweet["preprocessed_txt"] = str(doc.text).lower()

        operations.append(UpdateOne({"_id": parsed_tweet["_id"]}, {
            "$set": {"preprocessed_txt": parsed_tweet["preprocessed_txt"]}}, upsert=False))

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []
            settings.logger.debug("Progress %s out of %s", count, partition[1])

    if operations:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)


def fetch_es_tweets(connection_params, args):
    """ Scroll an elasticsearch instance and insert the tweets into MongoDB
    """

    db_name = connection_params[0]
    target_collection = connection_params[1]

    es_url = args[0]
    es_index = args[1]
    doc_type = args[2]
    field = args[3]
    lookup_list = args[4]
    _es = elasticsearch_base.connect(es_url)

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    es_results = elasticsearch_base.match(
        _es, es_index, doc_type, field, lookup_list)

    operations = []
    for doc in es_results:
        operations.append(InsertOne(doc["_source"]))

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


def run_fetch_es_tweets():
    """ Fetch tweets from elasticsearch
    """
    connection_params = ["twitter", "melvyn_hs_users"]
    lookup_list = file_ops.read_csv_file(
        'melvyn_hs_user_ids', settings.TWITTER_SEARCH_PATH)
    fetch_es_tweets(connection_params, [
                    "192.168.2.33", "tweets", "tweet", "user.id_str", lookup_list])


def start_feature_extraction():
    """Run operations"""
    job_list = [
        ["twitter_annotated_datasets", "NAACL_SRW_2016_features",
            "text", "test_conll", "conll"],
        ["dailystormer_archive", "d_stormer_documents", "article",
            "d_stormer_documents_conll", "conll"],
        ["twitter", "melvyn_hs_users", "text", "melvyn_hs_users_conll", "conll"],
        ["manchester_event", "tweets", "text", "tweets_conll", "conll"],
        ["inauguration", "tweets",  "text", "tweets_conll", "conll"],
        ["uselections", "tweets", "text", "tweets_conll", "conll"],
        ["unfiltered_stream_May17", "tweets",
            "text", "tweets_conll", "conll"],
        ["twitter", "tweets", "text", "tweets_conll", "conll"]
    ]

    for job in job_list:
        run_parallel_pipeline(
            job[0:5], feature_extraction_pipeline, [job[0] + "_" + job[3], "Prep conll format"])


def start_store_preprocessed_text():
    """ Start the job
    """
    job_list = [
        ["twitter_annotated_datasets", "NAACL_SRW_2016_features", "text"],
        ["twitter_annotated_datasets",
         "NLP_CSS_2016_expert_features", "text"],
        ["twitter_annotated_datasets", "crowdflower_features", "text"],
        ["dailystormer_archive", "d_stormer_documents", "article"],
        ["twitter", "melvyn_hs_users", "text"],
        ["manchester_event", "tweets", "text"],
        ["inauguration", "tweets", "text"],
        ["uselections", "tweets", "text"],
        ["twitter", "candidates_hs_exp6_combo_3_Mar_9813004", "text"],
        ["unfiltered_stream_May17", "tweets", "text"],
        ["twitter", "tweets", "text"]
    ]
    for job in job_list:
        run_parallel_pipeline(
            job, store_preprocessed_text, ["store_preprocessed_text", "Preprocess Text"])
