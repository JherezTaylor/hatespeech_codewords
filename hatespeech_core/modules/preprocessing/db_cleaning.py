# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with cleaning and reducing
the size of the initial MongoDB collection
"""

import plotly
from plotly.graph_objs import Scatter, Layout
from ..utils import settings
from ..utils import file_ops
from ..db import mongo_base
from ..db import mongo_data_filters
from ..db import mongo_complex


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
                             settings.OUTPUT_PATH, list(cursor))


@file_ops.do_cprofile
def run_get_top_k_users(connection_params, lang_list, field_name):
    """Test and print results of top k aggregation
    """

    cursor = mongo_complex.get_top_k_users(connection_params, lang_list,
                                           field_name, settings.USER_MENTIONS_LIMIT)
    file_ops.write_json_file("user_distribution",
                             settings.OUTPUT_PATH, list(cursor))


@file_ops.do_cprofile
def run_get_top_k_hashtags(connection_params, lang_list, field_name, k_value):
    """Test and print results of top k aggregation
    """

    cursor = mongo_complex.get_top_k_hashtags(
        connection_params, lang_list, field_name, settings.HASHTAG_LIMIT, k_value)

    file_ops.write_json_file("hashtag_distribution",
                             settings.OUTPUT_PATH, list(cursor))


def generate_bar_chart(chart_title):
    """Generate a plotly bar_chart
    """

    json_obj = file_ops.read_json_file("hashtag_dist_en", settings.OUTPUT_PATH)
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

    result = db_response
    print(result.modified_count)
    file_ops.send_job_completion(
        [time1, time2], ["run_retweet_removal", connection_params[0] + ": Retweet Removal"])


def run_create_indexes(connection_params):
    """Init indexes.

    Stage 2 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.create_indexes(connection_params)
    time2 = file_ops.time()

    file_ops.send_job_completion(
        [time1, time2], ["create_indexes", connection_params[0] + ": Index creation took"])


def run_field_removal(connection_params):
    """Start the field removal task.

    Stage 3 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    db_response = mongo_data_filters.field_removal(connection_params)
    time2 = file_ops.time()

    result = db_response
    print(result.modified_count)
    file_ops.send_job_completion(
        [time1, time2], ["field_removal", connection_params[0] + ": Field Removal took"])

    time1 = file_ops.time()
    db_response = mongo_data_filters.quoted_status_field_removal(
        connection_params)
    time2 = file_ops.time()

    result = db_response
    print(result.modified_count)
    file_ops.send_job_completion(
        [time1, time2], ["quoted_status_field_removal", connection_params[0] + ": Quoted_status Field Removal took"])


def run_language_trimming(connection_params, lang_list):
    """Start the language trimming task.

    Stage 4 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    db_response = mongo_data_filters.language_trimming(
        connection_params, lang_list)
    time2 = file_ops.time()

    result = db_response
    print(result.modified_count)
    file_ops.send_job_completion(
        [time1, time2], ["language_trimming", connection_params[0] + ": Language trimming took"])


def run_field_flattening(connection_params, depth, job_name, field_params):
    """Start the field flattening task.

    Stage 5 in preprocessing pipeline.
    """

    field_name = field_params[0]
    field_to_set = field_params[1]
    field_to_extract = field_params[2]

    time1 = file_ops.time()
    if len(field_params) == 3:
        mongo_data_filters.field_flattening_base(
            connection_params, depth, field_name, field_to_set, field_to_extract)
    if len(field_params) == 6:
        mongo_data_filters.field_flattening_complex(
            connection_params, depth, field_params)

    time2 = file_ops.time()
    file_ops.send_job_completion(
        [time1, time2], [str(job_name) + " ", connection_params[0] + " " + str(job_name)])


def run_parse_extended_tweet(connection_params, depth, job_name):
    """Start the field flattening task.

    Stage 6 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.parse_extended_tweet(connection_params, depth)
    time2 = file_ops.time()
    file_ops.send_job_completion(
        [time1, time2], [str(job_name) + " ", connection_params[0] + " " + str(job_name)])


def run_final_field_removal(connection_params, job_name):
    """Start the final field removal task.

    Stage 7 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.final_field_removal(connection_params)
    time2 = file_ops.time()
    file_ops.send_job_completion(
        [time1, time2], [str(job_name) + " ", connection_params[0] + " " + str(job_name)])


def run_clean_source_field(connection_params, job_name):
    """Start the clean source field task.

    Stage 8 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.clean_source_field(connection_params)
    time2 = file_ops.time()
    file_ops.send_job_completion(
        [time1, time2], [str(job_name) + " ", connection_params[0] + " " + str(job_name)])


def run_update_missing_text():
    """ Start the missing tweet text replacement job
    """

    connection_params = ["twitter_test", "tweets"]
    time1 = file_ops.time()
    db_response = mongo_base.update_missing_text(connection_params)
    time2 = file_ops.time()

    file_ops.send_job_completion(
        [time1, time2], ["update_missing_text", connection_params[0] + ": Replace missing text took " + str(
            db_response[0]) + "Not Found: " + str(db_response[1]) + "Total Requests: "
            + str(db_response[2])])


def preprocessing_pipeline():
    """ Handle DB operations"""

    job_names = ["hashtags", "entities.urls", "user_mentions", "media"]
    field_names = ["entities.hashtags", "entities.urls",
                   "entities.user_mentions", "entities.media"]
    fields_to_set = ["hashtags", "urls",
                     "user_mentions", "screen_name", "id_str", "media", "url", "type"]
    field_to_extract = [".text", ".expanded_url",
                        ".screen_name", ".id_str", ".media_url", ".type"]

    # lang_list = ['en', 'und', 'es', 'ru', 'pt',
    #              'it', 'ja', 'fr', 'de', 'ar', 'zh']

    connection_params = ["inauguration", "tweets"]
    # connection_params = ["inauguration_no_filter", "tweets"]

    # hashtag_args = [field_names[0], fields_to_set[0], field_to_extract[0]]
    # url_args = [field_names[1], fields_to_set[1], field_to_extract[1]]
    # user_mentions_args = [field_names[2], fields_to_set[2], fields_to_set[
    #     3], fields_to_set[4], field_to_extract[2], field_to_extract[3]]
    # media_args = [field_names[3], fields_to_set[5], fields_to_set[
    #     6], fields_to_set[7], field_to_extract[4], field_to_extract[5]]

    # # Remove retweets
    # run_retweet_removal(connection_params)

    # # Create Indexes
    # run_create_indexes(connection_params)

    # # Remove unwanted and redundant fields
    # run_field_removal(connection_params)

    # run_language_trimming(connection_params, ['en', 'und'])

    # # Hashtags
    # run_field_flattening(
    #     connection_params, "top_level", job_names[0], hashtag_args)

    # # Urls
    # run_field_flattening(
    #     connection_params, "top_level", job_names[1], url_args)

    # # User mentions
    # run_field_flattening(
    #     connection_params, "top_level", job_names[2], user_mentions_args)

    # # # Media
    # run_field_flattening(
    #     connection_params, "top_level", job_names[3], media_args)

    # # Quoted_status Hashtags
    # run_field_flattening(connection_params, "quoted_status",
    #                      job_names[0], hashtag_args)

    # # Quoted_status Urls
    # run_field_flattening(connection_params, "quoted_status",
    #                      job_names[1], url_args)

    # # Quoted_status User mentions
    # run_field_flattening(connection_params, "quoted_status",
    #                      job_names[2], user_mentions_args)

    # # Quoted_status Media
    # run_field_flattening(connection_params, "quoted_status",
    #                      job_names[3], media_args)

    # # Parse extended tweet
    # run_parse_extended_tweet(
    #     connection_params, "top_level", "Top Level Extended Tweet")
    # run_parse_extended_tweet(
    #     connection_params, "quoted_status", "Quoted Status Extended Tweet")

    # Remove final field set
    run_final_field_removal(connection_params, "Final Field Removal")

    # Clean source field
    run_clean_source_field(connection_params, "Clean Source Field")
