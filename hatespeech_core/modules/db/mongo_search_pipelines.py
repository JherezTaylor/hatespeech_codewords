# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module covers functions used for idenitifying a candidate set of tweets
for further analysis and preprocessing refinement.

HS Candidates
Porn/Spam Candidates
"""

import itertools
from collections import defaultdict
from pymongo import InsertOne, UpdateOne
from ..utils import settings
from ..utils import file_ops
from ..utils import text_preprocessing
from . import mongo_base


# @notifiers.do_cprofile
def select_porn_candidates(connection_params, filter_options, partition):
    """ Iterate the specified collection and store the ObjectId
    of documents that have been tagged as being pornographic in nature.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: query (dict): Query to execute.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
            4: porn_black_list (list): List of porn keywords.
            5: hs_keywords (list) HS corpus.
            6: black_list (list) Custom words to filter on.
        partition   (tuple): Contains skip and limit values.
    """

    # Setup args for mongo_base.finder() call
    client = mongo_base.connect()
    db_name = connection_params[0]
    connection_params.insert(0, client)

    query = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]
    porn_black_list = filter_options[4]
    hs_keywords = filter_options[5]

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    # Store the documents for our bulkwrite
    staging = []
    operations = []

    progress = 0
    cursor = mongo_base.finder(connection_params, query, False)
    for document in cursor:
        progress = progress + 1
        set_intersects = text_preprocessing.do_create_ngram_collections(
            document["text"].lower(), [porn_black_list, hs_keywords, None])

        # unigram_intersect = set_intersects[0]
        ngrams_intersect = set_intersects[1]

        if ngrams_intersect:
            staging.append(document)
        else:
            # No intersection, skip entry
            pass

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            settings.logger.debug(
                "Progress: %s", (progress * 100) / partition[1])
            for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


# @profile
# @notifiers.do_cprofile
def select_hs_candidates(connection_params, filter_options, partition):
    """ Iterate the specified collection and check for tweets that contain
    hatespeech keywords.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: query (dict): Query to execute.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
            4: porn_black_list (list): List of porn keywords.
            5: hs_keywords (list) HS corpus.
            6: black_list (list) Custom words to filter on.
        partition   (tuple): Contains skip and limit values.
    """

    # Setup args for mongo_base.finder() call
    client = mongo_base.connect()
    db_name = connection_params[0]
    connection_params.insert(0, client)

    query = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]
    porn_black_list = filter_options[4]
    hs_keywords = filter_options[5]
    black_list = filter_options[6]
    account_list = filter_options[7]

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    # Store the documents for our bulkwrite
    staging = []
    staging_ngram_freq = defaultdict(list)
    operations = []
    # Keep track of how often we match an ngram in our blacklist
    porn_black_list_counts = dict.fromkeys(porn_black_list, 0)
    new_blacklist_accounts = []

    progress = 0
    cursor = mongo_base.finder(connection_params, query, False)
    for document in cursor:
        progress = progress + 1
        set_intersects = text_preprocessing.do_create_ngram_collections(
            document["text"].lower(), [porn_black_list, hs_keywords, black_list])

        # unigram_intersect = set_intersects[0]
        ngrams_intersect = set_intersects[1]
        hs_keywords_intersect = set_intersects[2]
        # black_list_intersect = set_intersects[3]

        if not ngrams_intersect and document["user"]["screen_name"] not in account_list and hs_keywords_intersect:
            staging.append(document)

        # Here we want to keep track of how many times a user has text that matches
        # one of our porn ngrams. Users below the threshold will be processed.
        elif ngrams_intersect and document["user"]["screen_name"] not in account_list and hs_keywords_intersect:
            staging_ngram_freq[document["user"][
                "screen_name"]].append(document)
            for token in ngrams_intersect:
                porn_black_list_counts[token] += 1
        else:
            # No hs intersections, skip entry and update blacklist count
            for token in ngrams_intersect:
                porn_black_list_counts[token] += 1

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            settings.logger.debug(
                "Progress: %s", (progress * 100) / partition[1])
            for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass

    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)

    # Check for users with porn ngram frequencies below threshold
    # Note that the cursor has already been exhausted and this now
    # becomes a local disk operation
    staging = []
    operations = []
    for screen_name in staging_ngram_freq:
        # Consider users that don't appear frequently and stage them
        if len(staging_ngram_freq[screen_name]) < settings.PNGRAM_THRESHOLD:
            staging = staging + staging_ngram_freq[screen_name]
        else:
            new_blacklist_accounts.append(screen_name)

    if staging:
        for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)

    file_ops.write_json_file(
        'porn_ngram_hits', settings.OUTPUT_PATH, porn_black_list_counts)
    file_ops.write_csv_file("new_porn_account_filter",
                            settings.OUTPUT_PATH, new_blacklist_accounts)


# @notifiers.do_cprofile
def select_general_candidates(connection_params, filter_options, partition):
    """ Iterate the specified collection and store the ObjectId
    of documents that do not match any hs or pornographic keywords.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: query (dict): Query to execute.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
            4: porn_black_list (list): List of porn keywords.
            5: hs_keywords (list) HS corpus.
            6: black_list (list) Custom words to filter on.
        partition   (tuple): Contains skip and limit values.
    """

    # Setup args for mongo_base.finder() call
    client = mongo_base.connect()
    db_name = connection_params[0]
    connection_params.insert(0, client)

    query = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]
    porn_black_list = filter_options[4]
    hs_keywords = filter_options[5]
    black_list = filter_options[6]

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    # Store the documents for our bulkwrite
    staging = []
    operations = []

    progress = 0
    cursor = mongo_base.finder(connection_params, query, False)
    for document in cursor:
        progress = progress + 1
        set_intersects = text_preprocessing.do_create_ngram_collections(
            document["text"].lower(), [porn_black_list, hs_keywords, black_list])

        unigram_intersect = set_intersects[0]
        ngrams_intersect = set_intersects[1]
        hs_keywords_intersect = set_intersects[2]
        black_list_intersect = set_intersects[3]

        if not ngrams_intersect and (len(unigram_intersect) > 1) and not black_list_intersect and not hs_keywords_intersect:
            staging.append(document)
        else:
            # No intersection, skip entry
            pass

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            settings.logger.debug(
                "Progress: %s", (progress * 100) / partition[1])
            for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass

    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


def emotion_coverage_pipeline(connection_params, filter_options, partition):
    """ Iterate the specified collection get the emtotion coverage for each
    tweet. As a secondary function, create ngrams for the CrowdFlower dataset
    if the argument is passed.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: query (dict): Query to execute.
            1: create_ngrams (bool): Create ngrams or not.
            2: projection (str): Document field name.
        partition   (tuple): Contains skip and limit values.
    """

    # Setup args for mongo_base.finder() call
    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    connection_params.insert(0, client)
    query = filter_options[0]
    create_ngrams = filter_options[1]
    projection = filter_options[2]

    # Set skip limit values
    query["skip"] = partition[0]
    query["limit"] = partition[1]

    # Setup client object for bulk op
    bulk_client = mongo_base.connect()
    dbo = bulk_client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    operations = []
    staging = []
    progress = 0
    cursor = mongo_base.finder(connection_params, query, False)
    for document in cursor:
        progress = progress + 1
        staging.append(document)

        if len(staging) == 3000:
            settings.logger.debug(
                "Progress: %s", (progress * 100) / partition[1])
            for job in text_preprocessing.parallel_emotion_coverage(staging, projection):
                if job:
                    operations.append(job)
                else:
                    pass
            staging = []

        if create_ngrams:
            cleaned_result = text_preprocessing.clean_tweet_text(document[
                                                                 projection], True)
            xgrams = ([(text_preprocessing.create_ngrams(cleaned_result[0], i))
                       for i in range(1, 6)])
            operations.append(UpdateOne({"_id": document["_id"]}, {
                "$set": {"unigrams": xgrams[0], "bigrams": xgrams[1], "trigrams": xgrams[2],
                         "quadgrams": xgrams[3], "pentagrams": xgrams[4]}}, upsert=False))

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []

    if (len(staging) % 3000) != 0:
        for job in text_preprocessing.parallel_emotion_coverage(staging, projection):
            if job:
                operations.append(job)
            else:
                pass
    if operations:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)


def linear_test(connection_params, filter_options):
    """ Iterate the specified collection and store the ObjectId
    of documents that have been tagged as being subjective with a negative sentiment.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: target_collection (str): Name of output collection.
            1: subj_check (bool): Check text for subjectivity.
            2: sent_check (bool): Check text for sentiment.
            3: porn_black_list (list): List of porn keywords.
            4: hs_keywords (list) HS corpus.
            5: black_list (list) Custom words to filter on.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    target_collection = filter_options[0]
    subj_check = filter_options[1]
    sent_check = filter_options[2]
    porn_black_list = filter_options[3]
    hs_keywords = filter_options[4]
    black_list = filter_options[5]

    # Store the documents for our bulkwrite
    staging = []
    operations = []
    # Keep track of how often we match an ngram in our blacklist
    porn_black_list_counts = dict.fromkeys(porn_black_list, 0)

    cursor_count = dbo[collection].count()
    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}},
                                  {"text": 1, "created_at": 1, "coordinates": 1,
                                   "place": 1, "user": 1, "source": 1,
                                   "in_reply_to_user_id_str": 1}, no_cursor_timeout=True)
    for document in cursor:
        progress = progress + 1
        set_intersects = text_preprocessing.do_create_ngram_collections(
            document["text"].lower(), [porn_black_list, hs_keywords, black_list])

        # unigram_intersect = set_intersects[0]
        ngrams_intersect = set_intersects[1]
        hs_keywords_intersect = set_intersects[2]
        black_list_intersect = set_intersects[3]

        if not ngrams_intersect and not black_list_intersect and hs_keywords_intersect:
            staging.append(document)
        else:
            # No intersections, skip entry and update blacklist count
            for token in ngrams_intersect:
                porn_black_list_counts[token] += 1

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            settings.logger.debug(
                "Progress: %s", (progress * 100) / cursor_count)
            for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in text_preprocessing.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    if operations:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


def build_annotation_experiment(word, sample_size):
    """ Accepts a list of words and randomly samples from several datasets.
    """

    core_tweets = [
        ["twitter", "tweets"],
        ["uselections", "tweets"],
        ["inauguration", "tweets"],
        ["inauguration_no_filter", "tweets"],
        ["unfiltered_stream_May17", "tweets"],
        ["manchester_event", "tweets"]
    ]

    hate_corpus = [
        ["dailystormer_archive", "d_stormer_titles"],
        ["twitter", "melvyn_hs_users"]
    ]

    twitter_clean = {}
    twitter_hate = {}
    hate_community = {}

    for connection_params in hate_corpus:
        pipeline = [
            {"$match": {"$and": [{"tokens": {"$in": [word]}}, {
                "has_hs_keywords": False}]}},
            {"$project": {"preprocessed_txt": 1, "_id": 1}},
            {"$sample": {"size": sample_size}}]
        results = mongo_base.aggregate(connection_params, pipeline)
        for entry in results:
            hate_community[entry["_id"]] = {"database": connection_params[0],
                                            "collection": connection_params[1],
                                            "text": entry["preprocessed_txt"]}

    for connection_params in core_tweets:
        pipeline = [
            {"$match": {"$and": [{"tokens": {"$in": [word]}}, {
                "has_hs_keywords": False}]}},
            {"$project": {"preprocessed_txt": 1, "_id": 1}},
            {"$sample": {"size": sample_size}}]
        results = mongo_base.aggregate(connection_params, pipeline)
        for entry in results:
            twitter_clean[entry["_id"]] = {"database": connection_params[0],
                                           "collection": connection_params[1],
                                           "text": entry["preprocessed_txt"]}
        pipeline = [
            {"$match": {"$and": [{"tokens": {"$in": [word]}}, {
                "has_hs_keywords": True}]}},
            {"$project": {"preprocessed_txt": 1, "_id": 1}},
            {"$sample": {"size": sample_size}}]
        results = mongo_base.aggregate(connection_params, pipeline)
        for entry in results:
            twitter_hate[entry["_id"]] = {"database": connection_params[0],
                                          "collection": connection_params[1],
                                          "text": entry["preprocessed_txt"]}

    twitter_clean = dict(itertools.islice(twitter_clean.items(), sample_size))
    twitter_hate = dict(itertools.islice(twitter_hate.items(), sample_size))
    hate_community = dict(itertools.islice(
        hate_community.items(), sample_size))
    return [twitter_clean, twitter_hate, hate_community]
