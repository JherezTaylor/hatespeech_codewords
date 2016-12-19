# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Preprocessing module
"""

import multiprocessing
from modules.utils import settings
from modules.utils import file_ops
from modules.db import mongo_base
from modules.db import mongo_data_filters
from modules.db import mongo_complex
from modules.utils import twokenize
from joblib import Parallel, delayed
import plotly
from textblob import TextBlob
from plotly.graph_objs import Scatter, Layout


@file_ops.do_cprofile
def run_get_language_distribution(connection_params):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """

    lang_list = mongo_base.get_language_list(connection_params)
    cursor = mongo_base.get_language_distribution(
        connection_params, lang_list)

    file_ops.write_json_file("language_distribution",
                             settings.DATA_PATH, list(cursor))


@file_ops.do_cprofile
def run_get_top_k_users(connection_params, lang_list, field_name):
    """Test and print results of top k aggregation
    """

    cursor = mongo_complex.get_top_k_users(connection_params, lang_list,
                                           field_name, settings.USER_MENTIONS_LIMIT)
    file_ops.write_json_file("user_distribution",
                             settings.DATA_PATH, list(cursor))


@file_ops.do_cprofile
def run_get_top_k_hashtags(connection_params, lang_list, field_name, k_value):
    """Test and print results of top k aggregation
    """

    cursor = mongo_complex.get_top_k_hashtags(
        connection_params, lang_list, field_name, settings.HASHTAG_LIMIT, k_value)

    file_ops.write_json_file("hashtag_distribution",
                             settings.DATA_PATH, list(cursor))


def generate_bar_chart(chart_title):
    """Generate a plotly bar_chart
    """

    json_obj = file_ops.read_json_file("hashtag_dist_en", settings.DATA_PATH)
    data_x = []
    data_y = []
    for document in json_obj:
        data_x.append(document["hashtag"])
        data_y.append(document["count"])

    plotly.offline.plot({
        "data": [Scatter(x=data_x[0:10], y=data_y[0:10])],
        "layout": Layout(title=chart_title)
    })


def run_retweet_removal(connection_params):
    """Start the retweet removal task.

    Stage 1 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    db_response = mongo_data_filters.retweet_removal(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    result = db_response
    print result.modified_count

    print "%s function took %0.3f ms" % ("retweet_removal", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Retweet removal took " + str(time_diff) + " ms", result)
    print send_notification.content


def run_create_indexes(connection_params):
    """Init indexes.

    Stage 2 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.create_indexes(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print "%s function took %0.3f ms" % ("create_indexes", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Index creation took " + str(time_diff) + " ms", "Complete")
    print send_notification.content


def run_field_removal(connection_params):
    """Start the field removal task.

    Stage 3 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    db_response = mongo_data_filters.field_removal(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    result = db_response
    print result.modified_count

    print "%s function took %0.3f ms" % ("field_removal", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Field Removal took " + str(time_diff) + " ms", result)
    print send_notification.content

    time1 = file_ops.time()
    db_response = mongo_data_filters.quoted_status_field_removal(
        connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    result = db_response
    print result.modified_count

    print "%s function took %0.3f ms" % ("quoted_status_field_removal", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ":Quoted_status Field Removal took " + str(time_diff) + " ms", result)
    print send_notification.content


def run_language_trimming(connection_params):
    """Start the language trimming task.

    Stage 4 in preprocessing pipeline.
    """

    lang_list = ['en', 'und', 'es', 'ru', 'pt',
                 'it', 'ja', 'fr', 'de', 'ar', 'zh']

    time1 = file_ops.time()
    db_response = mongo_data_filters.language_trimming(
        connection_params, lang_list)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    result = db_response
    print result.modified_count

    print "%s function took %0.3f ms" % ("language_trimming", time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": Language trimming took " + str(time_diff) + " ms", result)
    print send_notification.content


def run_field_flattening(connection_params, job_name, field_params):
    """Start the field flattening task.

    Stage 5 in preprocessing pipeline.
    """

    field_name = field_params[0]
    field_to_set = field_params[1]
    field_to_extract = field_params[2]

    time1 = file_ops.time()
    if len(field_params) == 3:
        mongo_data_filters.field_flattening_base(
            connection_params, field_name, field_to_set, field_to_extract)
    else:
        mongo_data_filters.field_flattening_complex(
            connection_params, field_params)

    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print "%s function took %0.3f ms" % (job_name, time_diff)
    send_notification = file_ops.send_job_notification(
        settings.MONGO_SOURCE + ": " + job_name + str(time_diff) + " ms", "Complete")
    print send_notification.content


def runner():
    """ Handle DB operations"""

    job_names = ["hashtags", "entities.urls", "user_mentions", "media"]
    field_names = ["entities.hashtags", "entities.urls",
                   "entities.user_mentions", "entities.media"]
    fields_to_set = ["hashtags", "urls",
                     "user_mentions", "screen_name", "id_str", "media", "url", "type"]
    field_to_extract = [".text", ".expanded_url",
                        ".screen_name", ".id_str", ".media_url", ".type"]

    client = mongo_base.connect()
    connection_params = [client, "twitter", "tweets"]
    # connection_params = [client, "uselections", "tweets"]
    # connection_params = [client, "test_database", "random_sample"]

    hashtag_args = [field_names[0], fields_to_set[0], field_to_extract[0]]
    url_args = [field_names[1], fields_to_set[1], field_to_extract[1]]
    user_mentions_args = [field_names[2], fields_to_set[2], fields_to_set[
        3], fields_to_set[4], field_to_extract[2], field_to_extract[3]]
    media_args = [field_names[3], fields_to_set[5], fields_to_set[
        6], fields_to_set[7], field_to_extract[4], field_to_extract[5]]

    # Remove retweets
    # run_retweet_removal(connection_params)

    # Create Indexes
    run_create_indexes(connection_params)

    # Remove unwanted and redundant fields
    run_field_removal(connection_params)

    # run_language_trimming(connection_params)

    # # Hashtags
    # run_field_flattening(
    #     connection_params, job_names[0], hashtag_args)

    # # Urls
    # run_field_flattening(
    #     connection_params, job_names[1], url_args)

    # User mentions
    # run_field_flattening(
    #     connection_params, job_names[2], user_mentions_args)

    # # Media
    # run_field_flattening(
    #     connection_params, job_names[3], media_args)


def main():
    """
    Test functionality
    """
    runner()


if __name__ == "__main__":
    main()

    # test_get_language_distribution(client)
    # file_ops.filter_hatebase_categories()
    # generate_bar_chart()
    # mongo_complex.parse_undefined_lang(client, "twitter", "und_backup", "und")
    # test_file_operations()
    # test_get_language_collection(client)

    # mongo_base.filter_object_ids(client, "twitter", "tweets", ["und"], "collection_objectId")
    # mongo_base.create_lang_collection(client, "twitter", "tweets" "ru")
    # mongo_complex.get_hashtag_collection(client, "twitter", "hashtag_dist_und")

    # user_mentions_map_reduce(client, "twitter", "collection_ru")
    # hashtag_map_reduce(client, "twitter", "collection_ru", "hashtag_ru")
    # test_get_top_k_users(client, "twitter", ["ru"], USER_MENTIONS)
    # test_get_top_k_hashtags(client, "twitter", ["ru"], HASHTAGS, 20)
