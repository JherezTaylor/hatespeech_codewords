# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides methods to query the MongoDB instance
"""
from collections import Counter
from pprint import pprint
from langid.langid import LanguageIdentifier, model
from pymongo import MongoClient
from pymongo import errors
from pymongo import UpdateOne
from pymongo import InsertOne
from bson.son import SON
from bson.code import Code
from bson.objectid import ObjectId
from ..utils import constants
from ..utils import file_ops


def connect():
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    try:
        conn = MongoClient(constants.DB_URL)
        print "Connected to DB at " + constants.DB_URL + " successfully"
    except errors.ConnectionFailure, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


def get_language_list(connection_params):
    """Fetches a list of all the languages within the collection.

     Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

    Returns:
        list: List of languages within the tweet collection.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    distinct_lang = dbo[collection].distinct("lang")
    return file_ops.unicode_to_utf(distinct_lang)


def get_language_distribution(connection_params, lang_list):
    """Calculates the language distribution of tweets.

    Matches on languages in the passed list.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang_list   (list): List of languages to match on.

    Returns:
        list: A list of dicts with the distribution for each language in
        lang_list. For example:

        [ {"count": 2684813, "language": "es" } ]
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]

    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$group": {"_id": "$lang", "count": {"$sum": 1}}},
        {"$project": {"language": "$_id", "count": 1, "_id": 0}},
        {"$sort": SON([("count", -1), ("language", -1)])},
        {"$out": collection + "_lang_distribution"}
    ]

    dbo[collection].aggregate(pipeline)
    return dbo[collection + "_lang_distribution"].find({}, {"count": 1, "language": 1, "_id": 0})


def create_lang_collection(connection_params, lang):
    """Creates a new collection with only tweets matching the specified language.

    Outputs result to new collection named after the collection arg.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang        (str): Language to match on.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": lang}},
        {"$out": "collection_" + lang}
    ]
    dbo[collection].aggregate(pipeline)


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
                             constants.DATA_PATH, frequency)
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
                             constants.DATA_PATH, frequency)
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
    file_ops.write_json_file(collection, constants.DATA_PATH, list(cursor))


def get_object_ids(connection_params, lang_list, output_name):
    """Creates a to create a collection of object ids.

    Filters using the lang field. Also writes result to a json file.

    Args:
       connection_params   (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang_list   (list): List of languages to match on.
        output_name (str):  Name of the file to output to.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {"_id": 1}},
        {"$out": output_name}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)

    result = []
    cursor = dbo[output_name].find({})
    for document in cursor:
        result.append(str(document["_id"]))

    file_ops.write_json_file(output_name, constants.DATA_PATH, result)


def filter_by_language(connection_params, lang_list, output_name):
    """Bulk operation to remove tweets by the lang field.abs

    Filters by any lang not in lang_list. This should ideally be
    run directly through mongo shellfor large collections.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        lang_list   (list): List of languages to match on.
        output_name (str):  Name of the collection to store ids of non removed tweets.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    bulk = dbo[collection].initialize_unordered_bulk_op()

    bulk.find({"lang": {"$nin": lang_list}}).remove()
    result = bulk.execute()
    print "Finished remove operation"
    pprint(result)

    get_object_ids(connection_params, lang_list, output_name)


@file_ops.do_cprofile
def fetch_by_object_id(connection_params, object_id):
    """Fetches the specified object from the specified collection.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        object_id   (str): Object ID to fetch.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    cursor = dbo[collection].find({"_id": ObjectId(object_id)})
    return list(cursor)


def finder(connection_params, k_items):
    """Fetches k obects from the specified collection.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        k_items     (int): Number of items to retrieve.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    cursor = dbo[collection].find().limit(k_items)
    for document in cursor:
        pprint(document)
        pprint(str(document["_id"]))


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
        # Send once every 1000 in batch
        if len(operations) == 1000:
            dbo[collection].bulk_write(operations, ordered=False)
            operations = []

        # if doclang[0] == "en" and doclang[1] >= 0.70:
        #     print document["text"]

    if len(operations) > 0:
        dbo[collection].bulk_write(operations, ordered=False)

    print Counter(lang_dist)
    print reduce(lambda x, y: x + y, accuracy) / len(accuracy)


@file_ops.do_cprofile
def keyword_search(connection_params, keyword_list, lang_list):
    """Perform a text search with the provided keywords.

    We also preprocess the tweet text in order to avoid redundant operations.
    Outputs value to new collection.

    Args:
        connection_params (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.

        keyword_list    (list): List of keywords to search for.
        lang_list       (list): List of languages to match on.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    # Store the documents for our bulkwrite
    operations = []
    # Keep track of the tweets that we have already seen, keep distinct.
    seen_set = set()
    dbo = client[db_name]
    for search_query in keyword_list:
        # Run an aggregate search for each keyword, might get better performance
        # from running n keywords at a time, but I'm not sure.
        pipeline = [
            {"$match": {"$and": [{"$text": {"$search": search_query}},
                                 {"id_str": {"$nin": list(seen_set)}},
                                 {"lang": {"$in": lang_list}},
                                 {"retweet_count": 0}]}},
            {"$project": {"_id": 1, "id_str": 1, "text": 1, "id": 1, "timestamp": 1,
                          "lang": 1, "user.id_str": 1, "user.screen_name": 1, "user.location": 1}},
            {"$out": "temp_set"}
        ]
        dbo[collection].aggregate(pipeline, allowDiskUse=True)

        cursor = dbo["temp_set"].find({}, no_cursor_timeout=True)
        entities = cursor[:]

        print "Keyword:", search_query, "| Count:", cursor.count(), " | Seen:", len(seen_set)
        for document in entities:
            seen_set.add(document["id_str"])
            # Create a new field and add the preprocessed text to it
            operations.append(document)

            # document["vector"] = file_ops.preprocess_text(document["text"])
            # operations.append(InsertOne(document))

            # Send once every 1000 in batch
            if len(operations) == 1000:
                operations = file_ops.parallel_preprocess(operations)
                dbo["keyword_collection"].bulk_write(operations, ordered=False)
                operations = []

    if len(operations) > 0:
        operations = file_ops.parallel_preprocess(operations)
        dbo["keywords_collection"].bulk_write(operations, ordered=False)

    # Clean Up
    dbo["temp_set"].drop()


## DEPRECATED ##

# def filter_by_language(connection_params, lang_list, output_name):
#     """Aggregation pipeline to remove tweets with a lang field not in
#     lang_list. This should ideally be run directly through mongo shell
#     for large collections.

#     Args:
#         client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
#         db_name     (str): Name of database to query.
#         collection      (str): Name of collection to use.
#         lang_list   (list): List of languages to match on.
#         output_name (str): Name of the collection to store ids of non removed tweets.
#     """
    # client = connection_params[0]
    # db_name = connection_params[1]
    # collection = connection_params[2]
#     dbo = client[db_name]
#     bulk = dbo[collection].initialize_unordered_bulk_op()
#     count = 0

#     pipeline = [
#         {"$match": {"lang": {"$nin": lang_list}}},
#         {"$project": {"lang": 1, "_id": 1}},
#         {"$group": {
#             "_id": {
#                 "lang": "$lang",
#             },
#             "ids": {"$push": "$_id"}
#         }},
#         {"$project": {"ids": 1}}
#     ]
#     cursor = dbo[collection].aggregate(pipeline, allowDiskUse=True)
#     print "Finished aggregation. Iterating now"

#     for document in cursor:
#         bulk.find({"_id": {"$in": document["ids"]}}).remove()
#         count = count + 1
#         print "Count:", count

#         if count % 1000 == 0:
#             print "Running bulk execute"
#             bulk.execute()
#             bulk = dbo[collection].initialize_unordered_bulk_op()

#     if count % 1000 != 0:
#         print "Running bulk execute"
#         bulk.execute()

#     pipeline = [
#         {"$project": {"_id": 1}},
#         {"$out": output_name}
#     ]
#     dbo[collection].aggregate(pipeline, allowDiskUse=True)
