# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with idenitfy tweets that might be instances of
hatespeech
"""

from joblib import Parallel, delayed, cpu_count
from ..utils import settings
from ..utils import file_ops
from ..db import mongo_base
from ..db import mongo_search_pipelines


def run_select_porn_candidates(connection_params):
    """ Start the Porn indentification pipeline
    """

    client = mongo_base.connect()
    query = {}

    query["filter"] = {"text": {"$ne": None}}
    query["projection"] = {"text": 1, "created_at": 1, "coordinates": 1,
                           "place": 1, "user": 1, "source": 1, "in_reply_to_user_id_str": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    connection_params.insert(0, client)
    collection_size = mongo_base.finder(connection_params, query, True)
    del connection_params[0]
    client.close()

    num_cores = cpu_count()
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    # Load keywords once and avoid redundant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword
    porn_black_list = set(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    args2 = [query, "placeholder", False,
             False, porn_black_list, hs_keywords]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_porn_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_porn_candidates", connection_params[0] + ": Porn Candidates"])


def run_select_hs_candidates(connection_params):
    """ Start the HS indentification pipeline
    """

    client = mongo_base.connect()
    query = {}

    query["filter"] = {"text": {"$ne": None},
                       "urls_extracted": {"$exists": False}}
    query["projection"] = {"text": 1, "created_at": 1, "coordinates": 1,
                           "place": 1, "user": 1, "source": 1, "in_reply_to_user_id_str": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    connection_params.insert(0, client)
    collection_size = mongo_base.finder(connection_params, query, True)
    del connection_params[0]
    client.close()

    num_cores = cpu_count()
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    # Load keywords once and avoid redundant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword
    porn_black_list = set(file_ops.read_csv_file(
        "porn_trigrams_top_k_users", settings.WORDLIST_PATH))

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))
    black_list = set(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))
    account_list = set(file_ops.read_csv_file(
        "porn_account_filter", settings.WORDLIST_PATH))

    args2 = [query, "candidates_hs_exp6_combo_3_Mar", False, False,
             porn_black_list, hs_keywords, black_list, account_list]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_hs_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_hs_candidates", connection_params[0] + ": HS Candidates"])


def run_select_general_candidates(connection_params):
    """ Start the General indentification pipeline
    """

    client = mongo_base.connect()
    query = {}

    query["filter"] = {"text": {"$ne": None},
                       "urls_extracted": {"$exists": False}}
    query["projection"] = {"text": 1, "created_at": 1, "coordinates": 1,
                           "place": 1, "user": 1, "source": 1, "in_reply_to_user_id_str": 1}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    connection_params.insert(0, client)
    collection_size = mongo_base.finder(connection_params, query, True)
    del connection_params[0]
    client.close()

    num_cores = cpu_count()
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    # Load keywords once and avoid redundant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword

    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    hs_keywords = dict.fromkeys(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                                file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))
    black_list = dict.fromkeys(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))

    args2 = [query, "tester", False,
             False, porn_black_list, hs_keywords, black_list]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_general_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_gen_candidates", connection_params[0] + ": General Candidates"])


def send_job_completion(run_time, args):
    """Format and print the details of a completed job

    Args:
        run_time (list): Start and end times.
        args (list): Contains the following:
            0: function_name (str): Name of the function that was run.
            1: message_text  (str): Text to be sent in notification.
    """

    time_diff = round((run_time[1] - run_time[0]), 2)
    print("%s function took %0.3f seconds" % (args[0], time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": " + args[1] + " took " + str(time_diff) + " seconds", "Complete")
    print(send_notification.content)


def get_ngrams(connection_params, ngram_field):
    """Fetch the specified ngrams from the top k users that tweet porn/spam
    Args:
        ngram_field (str): Field to query on.
    """

    client = mongo_base.connect()
    connection_params.insert(0, client)
    ngram_set = set()
    user_accounts = set(file_ops.read_csv_file(
        "porn_account_filter", settings.WORDLIST_PATH))

    count = 0
    for account in user_accounts:
        count += 1
        print("Count: {0} out of {1}".format(count, len(user_accounts)))
        query = {}
        query["filter"] = {"user.screen_name": account}
        query["projection"] = {"_id": False, ngram_field: True}
        query["limit"] = 0
        cursor = mongo_base.finder(connection_params, query, False)

        # Extract the ngrams out of the object and flatten the 2D list
        xgrams = list(file_ops.chain.from_iterable(
            [doc[ngram_field] for doc in cursor]))
        ngram_set = ngram_set.union(xgrams)
    file_ops.write_csv_file('porn_trigrams_top_k_users',
                            settings.WORDLIST_PATH, ngram_set)


def sentiment_pipeline():
    """Handle sentiment analysis tasks"""
    connection_params = ["twitter", "tweets"]
    # get_ngrams(connection_params, "trigrams")
    # run_select_porn_candidates(connection_params)
    run_select_hs_candidates(connection_params)
    # run_select_general_candidates(connection_params)
