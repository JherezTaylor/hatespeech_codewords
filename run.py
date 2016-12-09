# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Preprocessing module
"""

from modules.utils import constants
from modules.utils import fileops
from modules.db import mongomethods
from joblib import Parallel, delayed
import multiprocessing
import plotly
from textblob import TextBlob
from plotly.graph_objs import Scatter, Layout


def test_file_operations():
    """
    Test previous methods
    """
    file_list = fileops.get_filenames(constants.JSON_PATH)
    fileops.extract_corpus(file_list)
    # num_entries = fileops.count_entries(file_list)
    # pprint(num_entries)
    res = fileops.read_csv_file("about_sexual_orientation_eng_pg1",
                                constants.CSV_PATH)
    res2 = fileops.build_query_string(res)
    print res2


def test_get_language_subset(client):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = ["en", "und", "es"]
    cursor = mongomethods.get_language_distribution(
        client, "twitter", lang_list)
    for document in cursor:
        print document


@fileops.do_cprofile
def test_get_language_distribution(client):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = mongomethods.get_language_list(client, "twitter")
    cursor = mongomethods.get_language_distribution(
        client, "twitter", lang_list)
    fileops.write_json_file("language_distribution",
                            constants.DATA_PATH, list(cursor))


def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """Test and print results of top k aggregation
    """
    cursor = mongomethods.get_top_k_users(client, db_name, lang_list,
                                          k_filter, constants.USER_MENTIONS_LIMIT)
    fileops.write_json_file("user_distribution",
                            constants.DATA_PATH, list(cursor))


@fileops.do_cprofile
def test_get_top_k_hashtags(client, db_name, lang_list, k_filter, k_value):
    """Test and print results of top k aggregation
    """
    cursor = mongomethods.get_top_k_hashtags(
        client, db_name, lang_list, k_filter, constants.HASHTAG_LIMIT, k_value)
    fileops.write_json_file("hashtag_distribution",
                            constants.DATA_PATH, list(cursor))


def generate_bar_chart(chart_title):
    """Generate a plotly bar_chart
    """
    json_obj = fileops.read_json_file("hashtag_dist_en", constants.DATA_PATH)
    data_x = []
    data_y = []
    for document in json_obj:
        data_x.append(document["hashtag"])
        data_y.append(document["count"])

    plotly.offline.plot({
        "data": [Scatter(x=data_x[0:10], y=data_y[0:10])],
        "layout": Layout(title=chart_title)
    })


def par_test(list_a):
    return fileops.preprocess_text(list_a)


@fileops.do_cprofile
def run_par(list_a):
    num_cores = multiprocessing.cpu_count()
    
    results = Parallel(n_jobs=num_cores)(delayed(fileops.preprocess_text)(i) for i in list_a)
    return results

@fileops.do_cprofile
def linear(list_a):
    results = []
    for x in list_a:
        results.append(fileops.preprocess_text(x))
    return results

def main():
    """
    Test functionality
    """
    # list_a = []
    # for x in range(0, 100000):
    #     list_a.append("RT @marcobonzanini: 11 just #NLP an example! :D http://example.com #NLP")
    # run_par(list_a)
    # linear(list_a)
    # print TextBlob("RT @marcobonzanini: 11 just #NLP an example! :D http://example.com #NLP").sentiment
    # hey = fileops.preprocess_text("RT @marcobonzanini: 11 just #NLP an example! :D http://example.com #NLP")
    # print hey
    client = mongomethods.connect()
    mongomethods.keyword_search(
        client, "twitter", ['yardie'], ['en'])

if __name__ == "__main__":
    main()

     # fileops.filter_hatebase_categories()
    
    # generate_bar_chart()
    # mongomethods.parse_undefined_lang(client, "twitter", "und_backup", "und")
    
    # test_file_operations()
    # test_get_language_distribution(client)
    # test_get_language_subset(client)

    # mongomethods.filter_object_ids(client, "twitter", "tweets", ["und"], "subset_objectId")
    # mongomethods.create_lang_subset(client, "twitter", "ru")
    # mongomethods.get_hashtag_collection(client, "twitter", "hashtag_dist_und")
    # mongomethods.find_one(client, "twitter", "subset_objectId")

    # user_mentions_map_reduce(client, "twitter", "subset_ru")
    # hashtag_map_reduce(client, "twitter", "subset_ru", "hashtag_ru")
    # test_get_top_k_users(client, "twitter", ["ru"], USER_MENTIONS)
    # test_get_top_k_hashtags(client, "twitter", ["ru"], HASHTAGS, 20)
