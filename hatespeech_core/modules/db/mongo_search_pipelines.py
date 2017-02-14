# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

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


@file_ops.do_cprofile
def select_hs_candidates(connection_params, filter_options):
    """ Iterate the specified collection and store the ObjectId
    of documents that have been tagged as being subjective with a negative sentiment.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: check_garbage (bool): Check for garbage tweet.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
    """

    # Load keywords once and avoid redudant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword

    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    black_list = dict.fromkeys(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))

    hs_keywords = dict.fromkeys(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]

    check_garbage = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]

    # Store the documents for our bulkwrite
    staging = []
    operations = []

    cursor_count = dbo[collection].count()
    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}},
                                  {"text": 1, "created_at": 1}, no_cursor_timeout=True)
    for document in cursor:
        progress = progress + 1

        if check_garbage:
            # Check if the text consists primarily of links, mentions and tags
            if file_ops.is_garbage(document["text"], settings.GARBAGE_TWEET_DIFF) is False:

                set_intersects = file_ops.do_create_ngram_collections(
                    document["text"], [porn_black_list, hs_keywords, black_list])

                unigram_intersect = set_intersects[0]
                ngrams_intersect = set_intersects[1]
                hs_keywords_intersect = set_intersects[2]
                black_list_intersect = set_intersects[3]

                # The tweet should not contain any blacklisted keywords, porn ngrams and
                # the porn unigrams should be <=3.
                # Finally, check if the tweet contains any hs keywords
                if not black_list_intersect and not ngrams_intersect and (len(unigram_intersect) <= 3) and hs_keywords_intersect:
                    staging.append(document)
                else:
                    # No intersections, skip entry
                    pass
            else:
                # Tweet is garbage
                pass

        # Don't check for garbage
        else:
            set_intersects = file_ops.do_create_ngram_collections(
                document["text"], [porn_black_list, hs_keywords, black_list])

            unigram_intersect = set_intersects[0]
            ngrams_intersect = set_intersects[1]
            hs_keywords_intersect = set_intersects[2]
            black_list_intersect = set_intersects[3]

            if not black_list_intersect and not ngrams_intersect and (len(unigram_intersect) <= 3) and hs_keywords_intersect:
                staging.append(document)
            else:
                # No intersection, skip entry
                pass

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            print "Progress: ", (progress * 100) / cursor_count, "%"
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


@file_ops.do_cprofile
def select_porn_candidates(connection_params, filter_options):
    """ Iterate the specified collection and store the ObjectId
    of documents that have been tagged as being pornographic in nature.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: check_garbage (bool): Check for garbage tweet.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
    """
    # Load keywords once and avoid redudant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword

    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    hs_keywords = dict.fromkeys(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]

    check_garbage = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]
    # Store the documents for our bulkwrite
    staging = []
    operations = []

    cursor_count = dbo[collection].count()
    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}},
                                  {"text": 1, "created_at": 1}, no_cursor_timeout=True)
    for document in cursor:
        progress = progress + 1

        if check_garbage:
            # Check if the text consists primarily of links, mentions and tags
            if file_ops.is_garbage(document["text"], settings.GARBAGE_TWEET_DIFF) is False:

                set_intersects = file_ops.do_create_ngram_collections(
                    document["text"], [porn_black_list, hs_keywords, None])

                # unigram_intersect = set_intersects[0]
                ngrams_intersect = set_intersects[1]

                if ngrams_intersect:
                    staging.append(document)
                else:
                    # No intersection, skip entry
                    pass
            else:
                # Tweet is garbage
                pass

        # Don't check for garbage
        else:
            set_intersects = file_ops.do_create_ngram_collections(
                document["text"], [porn_black_list, hs_keywords, None])

            # unigram_intersect = set_intersects[0]
            ngrams_intersect = set_intersects[1]

            if ngrams_intersect:
                staging.append(document)
            else:
                # No intersection, skip entry
                pass

        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(staging) == settings.BULK_BATCH_SIZE:
            print "Progress: ", (progress * 100) / cursor_count, "%"
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


@file_ops.do_cprofile
def select_general_candidates(connection_params, filter_options):
    """ Iterate the specified collection and store the ObjectId
    of documents that do not match any hs or pornographic keywords.

    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
        filter_options    (list): Contains a list of filter conditions as follows:
            0: check_garbage (bool): Check for garbage tweet.
            1: target_collection (str): Name of output collection.
            2: subj_check (bool): Check text for subjectivity.
            3: sent_check (bool): Check text for sentiment.
    """
    # Load keywords once and avoid redudant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword

    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    black_list = dict.fromkeys(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))

    hs_keywords = dict.fromkeys(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]

    check_garbage = filter_options[0]
    target_collection = filter_options[1]
    subj_check = filter_options[2]
    sent_check = filter_options[3]
    # Store the documents for our bulkwrite
    staging = []
    operations = []

    cursor_count = dbo[collection].count()
    progress = 0
    cursor = dbo[collection].find({"text": {"$ne": None}},
                                  {"text": 1, "created_at": 1}, no_cursor_timeout=True)
    for document in cursor:
        progress = progress + 1

        if check_garbage:
            # Check if the text consists primarily of links, mentions and tags
            if file_ops.is_garbage(document["text"], settings.GARBAGE_TWEET_DIFF) is False:

                set_intersects = file_ops.do_create_ngram_collections(
                    document["text"], [porn_black_list, hs_keywords, black_list])

                unigram_intersect = set_intersects[0]
                ngrams_intersect = set_intersects[1]
                hs_keywords_intersect = set_intersects[2]
                black_list_intersect = set_intersects[3]

                # No porn or hs intersect
                if not ngrams_intersect and (len(unigram_intersect) > 1) and not black_list_intersect and not hs_keywords_intersect:
                    staging.append(document)
                else:
                    # No intersection, skip entry
                    pass
            else:
                # Tweet is garbage
                pass

        # Don't check for garbage
        else:
            set_intersects = file_ops.do_create_ngram_collections(
                document["text"], [porn_black_list, hs_keywords, black_list])

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
            print "Progress: ", (progress * 100) / cursor_count, "%"
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
