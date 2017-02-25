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
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    collection_size = dbo[collection].count()
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

    args2 = ["candidates_porn_exp4_14_Feb", False,
             False, porn_black_list, hs_keywords]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_porn_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_porn_candidates", "Porn Candidates"])


def run_select_hs_candidates(connection_params):
    """ Start the HS indentification pipeline
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    collection_size = dbo[collection].count()
    num_cores = cpu_count()
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    # Load keywords once and avoid redundant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword
    porn_black_list = set(file_ops.read_csv_file("porn_bigrams", settings.WORDLIST_PATH) +
                          file_ops.read_csv_file("porn_trigrams", settings.WORDLIST_PATH) +
                          file_ops.read_csv_file("porn_quadgrams", settings.WORDLIST_PATH))

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))
    black_list = set(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))

    args2 = ["parallel_test", False, False,
             porn_black_list, hs_keywords, black_list]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_hs_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_hs_candidates", "HS Candidates"])


def run_select_general_candidates(connection_params):
    """ Start the General indentification pipeline
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]

    collection_size = dbo[collection].count()
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

    args2 = ["candidates_gen_exp3_15_Jan", False,
             False, porn_black_list, hs_keywords, black_list]
    time1 = file_ops.time()
    Parallel(n_jobs=num_cores)(delayed(mongo_search_pipelines.select_general_candidates)(
        connection_params, args2, partition) for partition in partitions)
    time2 = file_ops.time()
    send_job_completion(
        [time1, time2], ["select_gen_candidates", "General Candidates"])


def send_job_completion(run_time, args):
    """Format and print the details of a completed job

    Args:
        run_time (list): Start and end times.
        args (list): Contains the following:
            0: function_name (str): Name of the function that was run.
            1: message_text  (str): Text to be sent in notification.
    """

    time_diff = run_time[1] - run_time[0] * 1000.0
    print("%s function took %0.3f ms" % (args[0], time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": " + args[1] + " took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def sentiment_pipeline():
    """Handle sentiment analysis tasks"""
    connection_params = ["twitter", "tweets"]
    run_select_hs_candidates(connection_params)
    # run_select_porn_candidates(connection_params)
    # run_select_general_candidates(connection_params)
