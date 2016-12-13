# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Preprocessing module
"""

from modules.utils import constants
from modules.utils import file_ops
from modules.db import mongo_methods
from joblib import Parallel, delayed
import multiprocessing
import plotly
from textblob import TextBlob
from plotly.graph_objs import Scatter, Layout


def test_file_operations():
    """
    Test previous methods
    """
    file_list = file_ops.get_filenames(constants.JSON_PATH)
    file_ops.extract_corpus(file_list)
    # num_entries = file_ops.count_entries(file_list)
    # pprint(num_entries)
    res = file_ops.read_csv_file("about_sexual_orientation_eng_pg1",
                                constants.CSV_PATH)
    res2 = file_ops.build_query_string(res)
    print res2


def test_get_language_subset(client, subset):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = ["en", "und", "es"]
    cursor = mongo_methods.get_language_distribution(
        client, "twitter", subset, lang_list)
    for document in cursor:
        print document


@file_ops.do_cprofile
def test_get_language_distribution(client, subset):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = mongo_methods.get_language_list(client, "twitter", subset)
    cursor = mongo_methods.get_language_distribution(
        client, "twitter", subset, lang_list)
    file_ops.write_json_file("language_distribution",
                            constants.DATA_PATH, list(cursor))


def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """Test and print results of top k aggregation
    """
    cursor = mongo_methods.get_top_k_users(client, db_name, lang_list,
                                          k_filter, constants.USER_MENTIONS_LIMIT)
    file_ops.write_json_file("user_distribution",
                            constants.DATA_PATH, list(cursor))


@file_ops.do_cprofile
def test_get_top_k_hashtags(client, db_name, lang_list, k_filter, k_value):
    """Test and print results of top k aggregation
    """
    cursor = mongo_methods.get_top_k_hashtags(
        client, db_name, lang_list, k_filter, constants.HASHTAG_LIMIT, k_value)
    file_ops.write_json_file("hashtag_distribution",
                            constants.DATA_PATH, list(cursor))


def generate_bar_chart(chart_title):
    """Generate a plotly bar_chart
    """
    json_obj = file_ops.read_json_file("hashtag_dist_en", constants.DATA_PATH)
    data_x = []
    data_y = []
    for document in json_obj:
        data_x.append(document["hashtag"])
        data_y.append(document["count"])

    plotly.offline.plot({
        "data": [Scatter(x=data_x[0:10], y=data_y[0:10])],
        "layout": Layout(title=chart_title)
    })


def main():
    """
    Test functionality
    """
    data = file_ops.read_json_file('filter1_subset', constants.DATA_PATH)
    print data[0]
    # print data


    # client = mongo_methods.connect()
    # mongo_methods.keyword_search(
    #     client, "twitter", file_ops.parse_category_files(), ['en', 'und'])
    # print mongo_methods.finder(client, "twitter", "subset_ru", 1)
    # mongo_methods.subset_object_ids(client, "twitter", "tweets", ['en', 'und'], "subset_object_id")
    # mongo_methods.filter_by_language(client, "twitter", "ru_backup", ["en", "und", "es"], "lang_obj_ids")

if __name__ == "__main__":
    main()

    # test_get_language_distribution(client)
    # file_ops.filter_hatebase_categories()
    # generate_bar_chart()
    # mongo_methods.parse_undefined_lang(client, "twitter", "und_backup", "und")
    # test_file_operations()
    # test_get_language_subset(client)

    # mongo_methods.filter_object_ids(client, "twitter", "tweets", ["und"], "subset_objectId")
    # mongo_methods.create_lang_subset(client, "twitter", "ru")
    # mongo_methods.get_hashtag_collection(client, "twitter", "hashtag_dist_und")

    # user_mentions_map_reduce(client, "twitter", "subset_ru")
    # hashtag_map_reduce(client, "twitter", "subset_ru", "hashtag_ru")
    # test_get_top_k_users(client, "twitter", ["ru"], USER_MENTIONS)
    # test_get_top_k_hashtags(client, "twitter", ["ru"], HASHTAGS, 20)
