# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Preprocessing module
"""

from modules.utils import constants
from modules.utils import fileops
from modules.db import mongomethods


def test_file_operations():
    """
    Test previous methods
    """
    file_list = fileops.get_filenames(constants.JSON_PATH)
    fileops.extract_corpus(file_list)
    # num_entries = fileops.count_entries(file_list)
    # pprint(num_entries)
    res = fileops.read_csv_file('about_sexual_orientation_eng_pg1',
                                constants.CSV_PATH)
    res2 = fileops.build_query_string(res)
    print res2


def test_get_language_subset(client):
    """Test and print the results of aggregation
    Constrains language list to en, und, es.

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = ['en', 'und', 'es']
    cursor = mongomethods.get_language_distribution(
        client, 'twitter', lang_list)
    for document in cursor:
        print document


@fileops.do_cprofile
def test_get_language_distribution(client):
    """Test and print results of aggregation

    Args:
        client (pymongo.MongoClient): Connection object for Mongo DB_URL.
    """
    lang_list = mongomethods.get_language_list(client, 'twitter')
    cursor = mongomethods.get_language_distribution(
        client, 'twitter', lang_list)
    fileops.write_json_file('language_distribution',
                            constants.DATA_PATH, list(cursor))


def test_get_top_k_users(client, db_name, lang_list, k_filter):
    """Test and print results of top k aggregation
    """
    cursor = mongomethods.get_top_k_users(client, db_name, lang_list,
                                          k_filter, constants.USER_MENTIONS_LIMIT)
    fileops.write_json_file('user_distribution',
                            constants.DATA_PATH, list(cursor))


@fileops.do_cprofile
def test_get_top_k_hashtags(client, db_name, lang_list, k_filter, k_value):
    """Test and print results of top k aggregation
    """
    cursor = mongomethods.get_top_k_hashtags(
        client, db_name, lang_list, k_filter, constants.HASHTAG_LIMIT, k_value)
    fileops.write_json_file('hashtag_distribution',
                            constants.DATA_PATH, list(cursor))


def main():
    """
    Test functionality
    """

    client = mongomethods.connect()
    mongomethods.parse_undefined_lang(client, 'twitter', 'tweets', 'sd')


if __name__ == '__main__':
    main()

    # test_file_operations()
    # test_get_language_distribution(client)
    # test_get_language_subset(client)

    # mongomethods.filter_object_ids(client, 'twitter', 'tweets', ['und'], 'subset_objectId')
    # mongomethods.create_lang_subset(client, 'twitter', 'ru')
    # mongomethods.get_hashtag_collection(client, 'twitter', 'hashtag_dist_und')
    # mongomethods.find_one(client, 'twitter', 'subset_objectId')

    # user_mentions_map_reduce(client, 'twitter', 'subset_ru')
    # hashtag_map_reduce(client, 'twitter', 'subset_ru', 'hashtag_ru')
    # test_get_top_k_users(client, 'twitter', ['ru'], USER_MENTIONS)
    # test_get_top_k_hashtags(client, 'twitter', ['ru'], HASHTAGS, 20)
