# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module covers functions used for idenitifying a candidate set of tweets
for further analysis and preprocessing refinement.

HS Candidates
Porn/Spam Candidates
"""

from pymongo import InsertOne
from ..utils import settings
from ..utils import file_ops
from . import mongo_base


# @file_ops.do_cprofile
def select_porn_candidates(connection_params, filter_options, partition):
    """ Iterate the specified collection and store the ObjectId
    of documents that have been tagged as being pornographic in nature.

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
        partition   (tuple): Contains skip and limit values.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    target_collection = filter_options[0]
    subj_check = filter_options[1]
    sent_check = filter_options[2]
    porn_black_list = filter_options[3]
    hs_keywords = filter_options[4]

    # Store the documents for our bulkwrite
    staging = []
    operations = []

    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}},
                                  {"text": 1, "created_at": 1, "coordinates": 1,
                                   "place": 1, "user": 1, "source": 1,
                                   "in_reply_to_user_id_str": 1}, no_cursor_timeout=True).skip(partition[0]).limit(partition[1])
    for document in cursor:
        progress = progress + 1
        set_intersects = file_ops.do_create_ngram_collections(
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
            print("Progress: ", (progress * 100) / partition[1], "%")
            for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


# @profile
# @file_ops.do_cprofile
def select_hs_candidates(connection_params, filter_options, partition):
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
        partition   (tuple): Contains skip and limit values.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

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

    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}, "urls_extracted": {"$exists": False}},
                                  {"text": 1, "created_at": 1, "coordinates": 1,
                                   "place": 1, "user": 1, "source": 1,
                                   "in_reply_to_user_id_str": 1}, no_cursor_timeout=True).skip(partition[0]).limit(partition[1])
    for document in cursor:
        progress = progress + 1
        set_intersects = file_ops.do_create_ngram_collections(
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
            print("Progress: ", (progress * 100) / partition[1], "%")
            for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
    file_ops.write_json_file(
        'porn_ngram_hits', settings.DATA_PATH, porn_black_list_counts)


# @file_ops.do_cprofile
def select_general_candidates(connection_params, filter_options, partition):
    """ Iterate the specified collection and store the ObjectId
    of documents that do not match any hs or pornographic keywords.

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
        partition   (tuple): Contains skip and limit values.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    target_collection = filter_options[0]
    subj_check = filter_options[1]
    sent_check = filter_options[2]
    porn_black_list = filter_options[3]
    hs_keywords = filter_options[4]
    black_list = filter_options[5]

    # Store the documents for our bulkwrite
    staging = []
    operations = []

    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}, "urls_extracted": {"$exists": False}},
                                  {"text": 1, "created_at": 1, "coordinates": 1,
                                   "place": 1, "user": 1, "source": 1,
                                   "in_reply_to_user_id_str": 1}, no_cursor_timeout=True).skip(partition[0]).limit(partition[1])
    for document in cursor:
        progress = progress + 1
        set_intersects = file_ops.do_create_ngram_collections(
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
            print("Progress: ", (progress * 100) / partition[1], "%")
            for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass

    _ = mongo_base.do_bulk_op(dbo, target_collection, operations)


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
        set_intersects = file_ops.do_create_ngram_collections(
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
            print("Progress: ", (progress * 100) / cursor_count, "%")
            for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
                if job:
                    operations.append(InsertOne(job))
                else:
                    pass
            staging = []

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(staging) % settings.BULK_BATCH_SIZE) != 0:
        for job in file_ops.parallel_preprocess(staging, hs_keywords, subj_check, sent_check):
            if job:
                operations.append(InsertOne(job))
            else:
                pass
    _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
