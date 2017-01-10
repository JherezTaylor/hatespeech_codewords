# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module calls functions that deal with idenitfy tweets that might be instances of
hatespeech
"""

from ..utils import settings
from ..utils import file_ops
from ..db import mongo_base
from ..db import mongo_complex


def run_select_hs_candidates(connection_params):
    """ Start the HS indentification pipeline
    """

    time1 = file_ops.time()
    mongo_complex.select_hs_candidates(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print "%s function took %0.3f ms" % ("select_hs_candidates", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": HS select candidates took " + str(time_diff) + " ms", "Complete")
    print send_notification.content


def sentiment_pipeline():
    """Handle sentiment analysis tasks"""
    client = mongo_base.connect()
    connection_params = [client, "twitter_test", "tweets"]
    run_select_hs_candidates(connection_params)

    # TODO Make this a test case
    # doc = {}
    # doc["text"] = "I sex predict &amp; :D RT rt I won't win a single game I bet on. Got Cliff Lee today, so if he loses its on me RT @e_one: Texas (cont) http://tl.gd/6meogh"
    # print file_ops.preprocess_tweet(doc, doc["text"].split(), black_list, hs_keywords)
    # print file_ops.preprocess_text(doc["text"])
    # black_list = dict.fromkeys(file_ops.read_csv_file(
    #     "blacklist", settings.WORDLIST_PATH))

    # hs_keywords = dict.fromkeys(file_ops.read_csv_file("hate_1", settings.TWITTER_SEARCH_PATH) +
    #                             file_ops.read_csv_file("hate_2", settings.TWITTER_SEARCH_PATH) +
    # file_ops.read_csv_file("hate_3", settings.TWITTER_SEARCH_PATH))
