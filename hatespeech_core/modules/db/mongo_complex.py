# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides complex methods that interact with the MongoDB instance.

Aggregate users and hashtags.
Map reduce samples.
Keyword searches. Only provided as a sample, for large collections ElasticSearch should be used.
Identify the language of tweets with an und lang.
"""

from collections import Counter
from pprint import pprint
from bson.code import Code
from bson.son import SON
from bson.objectid import ObjectId
from langid.langid import LanguageIdentifier, model
from pymongo import UpdateOne, InsertOne
from ..utils import settings
from ..utils import file_ops
from . import mongo_base


def get_top_k_users(connection_params, lang_list, field_name, k_limit):
    """Finds the top k users in the collection.

    This function outputs the aggregated results to a new collection in the db. From this
    we return the top k results.
    field_name is the name of an array in the collection, we apply the $unwind operator to it

     Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang_list   (list): List of languages to match on.
        field_name  (str):  Name of an array in the tweet object.
        k_limit     (int):  Limit for the number of results to return.

    Returns:
        list: List of dicts containing id_str, screen_name and the frequency of
        appearance. For example:

        [ {"count": 309, "id_str": "3407098175", "screen_name": "CharmsPRru" }]
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    field_name_base = field_name
    field_name = "$" + field_name
    dbo = client[db_name]

    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {field_name_base: 1, "_id": 0}},
        {"$unwind": field_name},
        {"$group": {"_id": {"id_str": field_name + ".id_str", "screen_name":
                            field_name + ".screen_name"}, "count": {"$sum": 1}}},
        {"$project": {"id_str": "$_id.id_str",
                      "screen_name": "$_id.screen_name", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("id_str", -1)])},
        {"$out": collection + "_top_k_users"}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)
    return dbo[collection + "_top_k_users"].find({}, {"count": 1, "screen_name": 1,
                                                      "id_str": 1, "_id": 0}).limit(k_limit)


def get_top_k_hashtags(connection_params, lang_list, field_name, k_limit, k_value):
    """Finds the top k hashtags in the collection.
    field_name is the name of an array in the collection, we apply the $unwind operator to it

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        field_name    (str):  Name of an array in the collection.abs
        limit       (int):  Limit for the number of results to return.
        k_value     (int):  Filter for the number of occurences for each hashtag

    Returns:
        list: List of dicts containing id_str, screen_name and the frequency of
        appearance. For example:

        [ {"count": 309, "hashtag": "cats" } ]
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    field_name_base = field_name
    field_name = "$" + field_name
    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {field_name_base: 1, "_id": 0}},
        {"$unwind": field_name},
        {"$group": {"_id": field_name + ".text", "count": {"$sum": 1}}},
        {"$project": {"hashtag": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("_id", -1)])},
        {"$match": {"count": {"$gt": k_value}}},
        {"$out": collection + "_top_k_hashtags"}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)
    return dbo[collection + "_top_k_hashtags"].find({}, {"count": 1, "hashtag": 1,
                                                         "_id": 0}).limit(k_limit)


def user_mentions_map_reduce(connection_params, output_name):
    """Map reduce that returns the number of times a user is mentioned.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        output_name (str): Name of the file to output to.

    Returns:
        list: List of objects containing user_screen_name and the frequency of appearance.
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
    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    cursor = dbo[collection].map_reduce(
        map_function, reduce_function, output_name, query={"lang": {"$eq": "en"}})

    for document in cursor.find():
        frequency.append({"_id": document["_id"], "value": document["value"]})

    frequency = sorted(frequency, key=lambda k: k["value"], reverse=True)
    file_ops.write_json_file("user_distribution_mr",
                             settings.DATA_PATH, frequency)
    pprint(frequency)


def hashtag_map_reduce(connection_params, output_name):
    """Map reduce that returns the number of times a hashtag is used.

    Args:
       connection_params   (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        output_name (str): Name of the file to output to.

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
    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    cursor = dbo[collection].map_reduce(
        map_function, reduce_function, output_name, query={"lang": {"$eq": "en"}})

    for document in cursor.find():
        frequency.append({"_id": document["_id"], "value": document["value"]})

    frequency = sorted(frequency, key=lambda k: k["value"], reverse=True)
    file_ops.write_json_file("hashtag_distribution_mr",
                             settings.DATA_PATH, frequency)
    pprint(frequency)


def fetch_hashtag_collection(connection_params):
    """Fetches the specified hashtag collection and writes it to a json file

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    cursor = dbo[collection].find({"count": {"$gt": 500}}, {
        "hashtag": 1, "count": 1, "_id": 0})
    file_ops.write_json_file(collection, settings.DATA_PATH, list(cursor))


@file_ops.do_cprofile
def parse_undefined_lang(connection_params, lang):
    """Parse the text of each tweet and identify and update its language.

    Be careful when using this, most of the tweets marked as und are composed
    mostly of links and hashtags, which might be useful. Also consider the confidence
    score.

    Args:
       connection_params   (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang        (str): Language to match on.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    # Normalize the confidence value with the range 0 - 1
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    # Operation statistics
    operations = []
    lang_dist = []
    accuracy = []

    cursor = dbo[collection].find(
        {"lang": lang}, {"text": 1, "lang": 1, "_id": 1}, no_cursor_timeout=True)
    for document in cursor:

        doclang = identifier.classify(document["text"])
        lang_dist.append(doclang[0])
        accuracy.append(doclang[1])

        if doclang[0] == "en" and doclang[1] >= 0.70:
            operations.append(
                UpdateOne({"_id": document["_id"]}, {
                    "$set": {"lang": "en"}})
            )
        # Send once every settings.BULK_BATCH_SIZE in batch
        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []

        # if doclang[0] == "en" and doclang[1] >= 0.70:
        #     print document["text"]

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)

    print Counter(lang_dist)
    print reduce(lambda x, y: x + y, accuracy) / len(accuracy)
