# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with idenitfy tweets that might be instances of
hatespeech
"""

from ..utils import settings
from ..utils import file_ops
from ..db import mongo_base
from ..db import mongo_search_pipelines


def run_select_porn_candidates(connection_params):
    """ Start the Porn indentification pipeline
    """

    # Load keywords once and avoid redudant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword
    porn_black_list = set(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))

    args2 = ["candidates_porn_exp4_14_Feb", False,
             False, porn_black_list, hs_keywords]
    time1 = file_ops.time()
    mongo_search_pipelines.select_porn_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_porn_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Porn Candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def run_select_hs_candidates(connection_params):
    """ Start the HS indentification pipeline
    """

    # Load keywords once and avoid redudant disk reads
    # Load our blacklist and filter any tweet that has a matching keyword
    porn_black_list = set(file_ops.read_csv_file("porn_bigrams", settings.WORDLIST_PATH) +
                          file_ops.read_csv_file("porn_trigrams", settings.WORDLIST_PATH) +
                          file_ops.read_csv_file("porn_quadgrams", settings.WORDLIST_PATH))

    hs_keywords = set(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
                      file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))
    black_list = set(file_ops.read_csv_file(
        "blacklist", settings.WORDLIST_PATH))

    args2 = ["candidates_hs_exp5_19_Feb", False, False,
             porn_black_list, hs_keywords, black_list]
    time1 = file_ops.time()
    mongo_search_pipelines.select_hs_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_hs_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": HS Candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def run_select_general_candidates(connection_params):
    """ Start the General indentification pipeline
    """

    # Load keywords once and avoid redudant disk reads
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
    mongo_search_pipelines.select_general_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_gen_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": General Candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def sentiment_pipeline():
    """Handle sentiment analysis tasks"""
    client = mongo_base.connect()
    connection_params = [client, "twitter", "tweets"]
    run_select_hs_candidates(connection_params)
    # run_select_porn_candidates(connection_params)
    # run_select_general_candidates(connection_params)
