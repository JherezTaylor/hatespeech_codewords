# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides methods to query the MongoDB instance
"""
from pprint import pprint
import pymongo
from bson.son import SON
from bson.code import Code
from modules.utils import constants
from modules.utils import fileops


def connect():
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    try:
        conn = pymongo.MongoClient(constants.DB_URL)
        print "Connected to DB at " + constants.DB_URL + " successfully"
    except pymongo.errors.ConnectionFailure, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


def get_language_list(client, db_name):
    """Returns a list of all the matching languages within the collection
     Args:
        client  (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name (str): Name of database to query.

    Returns:
        list: List of languages within the twitter collection.
    """
    dbo = client[db_name]
    distinct_lang = dbo.tweets.distinct("lang")
    return fileops.unicode_to_utf(distinct_lang)


def get_language_distribution(client, db_name, lang_list):
    """Returns the distribution of tweets matching either
    english, undefined or spanish.

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.

    Returns:
        list: Distribution for each language in lang_list.
    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
        {"$project": {"language": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("language", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline)


def create_lang_subset(client, db_name, lang):
    """Subsets the collection by the specified language.
    Outputs value to new collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang        (list): language to match on.

    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": lang}},
        {"$out": "subset_" + lang}
    ]
    dbo.tweets.aggregate(pipeline)


def get_top_k_users(client, db_name, lang_list, k_filter, limit):
    """Finds the top k users in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

     Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.

    Returns:
        list: List of objects containing id_str, screen_name and the frequency of appearance.
    """
    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": {"id_str": k_filter + ".id_str", "screen_name":
                            k_filter + ".screen_name"}, "count": {"$sum": 1}}},
        {"$project": {"id_str": "$_id.id_str",
                      "screen_name": "$_id.screen_name", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("id_str", -1)])}
    ]
    return dbo.tweets.aggregate(pipeline, allowDiskUse=True)


def get_top_k_hashtags(client, db_name, lang_list, k_filter, limit, k_value):
    """Finds the top k hashtags in the collection.
    k_filter is the name of an array in the collection, we apply the $unwind operator to it

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        k_filter    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.
        k_value     (int):  Filter for the number of occurences for each hashtag

    Returns:
        list: List of objects containing _id, hashtag text and the frequency of appearance.
    """

    k_filter_base = k_filter
    k_filter = "$" + k_filter
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {k_filter_base: 1, "_id": 0}},
        {"$unwind": k_filter},
        {"$group": {"_id": k_filter + ".text", "count": {"$sum": 1}}},
        {"$project": {"hashtag": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("_id", -1)])},
        {"$match": {"count": {"$gt": k_value}}},
        {"$limit": limit},
    ]
    return dbo.tweets.aggregate(pipeline)


def user_mentions_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a user is mentioned

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var userMentions = this.entities.user_mentions;"
                        "    for (var i = 0; i < userMentions.length; i ++){"
                        "        if (userMentions[i].screen_name.length > 0) {"
                        "            emit (userMentions[i].screen_name, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyUsername, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    fileops.write_json_file('user_distribution_mr',
                            constants.DATA_PATH, frequency)
    pprint(frequency)


def hashtag_map_reduce(client, db_name, subset, output_name):
    """Map reduce that returns the number of times a hashtag is used

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """
    map_function = Code("function () {"
                        "    var hashtags = this.entities.hashtags;"
                        "    for (var i = 0; i < hashtags.length; i ++){"
                        "        if (hashtags[i].text.length > 0) {"
                        "            emit (hashtags[i].text, 1);"
                        "        }"
                        "    }"
                        "}")

    reduce_function = Code("function (keyHashtag, occurs) {"
                           "     return Array.sum(occurs);"
                           "}")
    frequency = []
    dbo = client[db_name]
    cursor = dbo[subset].map_reduce(
        map_function, reduce_function, output_name)

    for document in cursor.find():
        frequency.append({'_id': document['_id'], 'value': document['value']})

    frequency = sorted(frequency, key=lambda k: k['value'], reverse=True)
    fileops.write_json_file('hashtag_distribution_mr',
                            constants.DATA_PATH, frequency)
    pprint(frequency)


def collection_finder(client, db_name, subset):
    """Fetches the specified collection
    """
    dbo = client[db_name]
    # cursor = dbo[subset].find({"count":{"$gt":500}})
    # cursor = dbo[subset].find(
    #     {}, no_cursor_timeout=True)
    # cursor.batch_size(80000)
    fileops.write_json_file(subset, constants.DATA_PATH, list(
        dbo[subset].find({}, {"hashtag": 1, "count": 1, "_id": 0})))

# def parse_undefined_lang(client, db_name, subset):

@fileops.do_cprofile
def find_one(client, db_name, subset):
    """Fetches one object from the specified collection
    """
    dbo = client[db_name]
    cursor = dbo[subset].find_one()
    pprint(cursor)
