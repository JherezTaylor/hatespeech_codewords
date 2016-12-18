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


def test_file_operations():
    """
    Test previous methods
    """

    file_list = file_ops.get_filenames(settings.JSON_PATH)
    file_ops.extract_corpus(file_list)
    # num_entries = file_ops.count_entries(file_list)
    # pprint(num_entries)
    res = file_ops.read_csv_file("about_sexual_orientation_eng_pg1",
                                 settings.CSV_PATH)
    res2 = file_ops.build_query_string(res)
    print res2


def test_get_language_collection(connection_params):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """

    lang_list = ["en", "und", "es"]
    cursor = mongo_base.get_language_distribution(
        connection_params, lang_list)

    result = list(cursor)
    for document in cursor:
        print document

    file_ops.send_job_notification(
        "test_get_language_collection", result)


@file_ops.do_cprofile
def test_get_language_distribution(connection_params):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """

    lang_list = mongo_base.get_language_list(connection_params)
    cursor = mongo_base.get_language_distribution(
        connection_params, lang_list)

    file_ops.write_json_file("language_distribution",
                             settings.DATA_PATH, list(cursor))


def test_get_top_k_users(connection_params, lang_list, field_name):
    """Test and print results of top k aggregation
    """

    cursor = mongo_complex.get_top_k_users(connection_params, lang_list,
                                           field_name, settings.USER_MENTIONS_LIMIT)
    file_ops.write_json_file("user_distribution",
                             settings.DATA_PATH, list(cursor))


@file_ops.do_cprofile
def test_get_top_k_hashtags(connection_params, lang_list, field_name, k_value):
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

def run_create_indexes(connection_params):
    """Init indexes.

    Stage 1 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    mongo_data_filters.create_indexes(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    print "%s function took %0.3f ms" % ("field_removal", time_diff)
    send_notification = file_ops.send_job_notification(
        "Field Removal took " + str(time_diff) + " ms", "None")
    print send_notification.content

def run_field_removal(connection_params):
    """Start the field removal task.

    Stage 2 in preprocessing pipeline.
    """

    time1 = file_ops.time()
    db_response = mongo_data_filters.field_removal(connection_params)
    time2 = file_ops.time()
    time_diff = (time2 - time1) * 1000.0

    result = db_response
    print result.modified_count

    print "%s function took %0.3f ms" % ("field_removal", time_diff)
    send_notification = file_ops.send_job_notification(
        "Field Removal took " + str(time_diff) + " ms", result)
    print send_notification.content


def main():
    """
    Test functionality
    """
    client = mongo_base.connect()
    connection_params = [client, "twitter", "tweets"]

    run_create_indexes(connection_params)
    run_field_removal(connection_params)
    # data = file_ops.read_json_file('filter1_collection', settings.DATA_PATH)
    # print data[0]
    # print data

    # client = mongo_base.connect()
    # connection_params = [client, "twitter", "tweets"]
    # test_get_language_collection(connection_params)
    # # test_get_language_distribution(connection_params)
    # temp = file_ops.get_filenames(settings.JSON_PATH)
    # print temp
    # print twokenize.tokenize("hey hi, o!")

    # r = file_ops.send_job_notification("hey", ["test10"])
    # print r
    # print r.status_code
    # print mongo_complex.get_language_list([client, "twitter", "tweets"])
    # test_get_language_distribution([client, "twitter", "tweets"], ["en", "und"])
    # mongo_complex.keyword_search(
    #     client, "twitter", file_ops.parse_category_files(), ['en', 'und'])
    # print mongo_complex.finder(client, "twitter", "collection_ru", 1)
    # mongo_complex.collection_object_ids(client, "twitter", "tweets", ['en', 'und'], "collection_object_id")
    # mongo_complex.filter_by_language(client, "twitter", "ru_backup", ["en", "und", "es"], "lang_obj_ids")

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
