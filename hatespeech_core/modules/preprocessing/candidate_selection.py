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


def run_select_hs_candidates(connection_params):
    """ Start the HS indentification pipeline
    """
    # args = [True, "candidates_hs_exp3_nogarb_15_Jan"]
    # time1 = file_ops.time()
    # mongo_search_pipelines.select_hs_candidates(connection_params, args)
    # time2 = file_ops.time()
    # time_diff = (time2 - time1) * 1000.0

    # print "%s function took %0.3f ms" % ("select_hs_candidates", time_diff)
    # send_notification = file_ops.send_job_notification(
    #     settings.MONGO_SOURCE + ": HS select candidates took " + str(time_diff) + " ms", "Complete")
    # print send_notification.content

    # Prep a collection with check garbage set to False
    # Garbage check seems to have no effect
    args2 = [False, "candidates_hs_exp5_19_Feb", False, False]
    time1 = file_ops.time()
    mongo_search_pipelines.select_hs_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_hs_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": HS select candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def run_select_porn_candidates(connection_params):
    """ Start the Porn indentification pipeline
    """
    # args = [True, "candidates_porn_exp3_nogarb_15_Jan", False, False]
    # time1 = file_ops.time()
    # mongo_search_pipelines.select_porn_candidates(connection_params, args)
    # time2 = file_ops.time()
    # time_diff = (time2 - time1) * 1000.0

    # print "%s function took %0.3f ms" % ("select_porn_candidates", time_diff)
    # send_notification = file_ops.send_job_notification(
    #     settings.MONGO_SOURCE + ": Porn select candidates took " + str(time_diff) + " ms", "Complete")
    # print send_notification.content

    # Prep a collection with check garbage set to False
    args2 = [False, "candidates_porn_exp4_14_Feb", False, False]
    time1 = file_ops.time()
    mongo_search_pipelines.select_porn_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_porn_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Porn select candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def run_select_general_candidates(connection_params):
    """ Start the General indentification pipeline
    """
    args = [True, "candidates_gen_exp3_nogarb_15_Jan"]
    time1 = file_ops.time()
    mongo_search_pipelines.select_general_candidates(connection_params, args)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_gen_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": General select candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)

    # Prep a collection with check garbage set to False
    args2 = [False, "candidates_gen_exp3_garb_15_Jan"]
    time1 = file_ops.time()
    mongo_search_pipelines.select_general_candidates(connection_params, args2)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print("%s function took %0.3f ms" % ("select_gen_candidates", time_diff))
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": General select candidates took " + str(time_diff) + " ms", "Complete")
    print(send_notification.content)


def sentiment_pipeline():
    """Handle sentiment analysis tasks"""
    client = mongo_base.connect()
    connection_params = [client, "twitter", "tweets"]
    run_select_hs_candidates(connection_params)
    # run_select_porn_candidates(connection_params)
    # run_select_general_candidates(connection_params)
