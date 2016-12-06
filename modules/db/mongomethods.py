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
from modules.utils import constants
from modules.utils import fileops


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
        output_name (str): Name of the collection to output to.

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
        output_name (str): Name of the collection to output to.

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


def get_hashtag_collection(client, db_name, subset):
    """Fetches the specified hashtag collection and writes it to a json file

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find({"count": {"$gt": 500}}, {
                              "hashtag": 1, "count": 1, "_id": 0})
    fileops.write_json_file(subset, constants.DATA_PATH, list(cursor))


def filter_object_ids(client, db_name, subset, lang_list, output_name):
    """Aggregation pipeline to filter object ids based on the provided condition

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        lang_list   (list): List of languages to match on.
        output_name (str): Name of the collection to output to.

    Returns:
        list: List of objects containing _id and the frequency of appearance.
    """

    dbo = client[db_name]
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {"_id": 1}},
        {"$out": output_name}
    ]
    dbo[subset].aggregate(pipeline, allowDiskUse=True)

    result = []
    cursor = dbo[output_name].find({})
    for document in cursor:
        result.append(str(document['_id']))

    fileops.write_json_file(output_name, constants.DATA_PATH, result)


def find_by_object_id(client, db_name, subset, object_id):
    """Fetches the specified object from the specified collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        object_id   (str): Object ID to fetch.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find({"_id": ObjectId(object_id)})
    pprint(cursor['_id'])


def finder(client, db_name, subset, k_items):
    """Fetches k obects from the specified collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        k_items     (int): Number of items to retrieve.
    """
    dbo = client[db_name]
    cursor = dbo[subset].find().limit(k_items)
    for document in cursor:
        pprint(document)
        pprint(str(document['_id']))


@fileops.do_cprofile
def parse_undefined_lang(client, db_name, subset, lang):
    """Parse the text of each tweet and identify and update its language
    Be careful when using this, most of the tweets marked as und are composed
    mostly of links and hashtags, which might be useful. Also consider the confidence
    score.

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str): Name of database to query.
        subset      (str): Name of collection to use.
        lang        (str): Language to match on.
    """

    dbo = client[db_name]
    # Normalize the confidence value with the range 0 - 1
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

    # Operation statistics
    operations = []
    lang_dist = []
    accuracy = []

    for document in dbo[subset].find({'lang': lang}, no_cursor_timeout=True):

        doclang = identifier.classify(document['text'])
        lang_dist.append(doclang[0])
        accuracy.append(doclang[1])

        if doclang[0] == 'en' and doclang[1] >= 0.70:
            operations.append(
                UpdateOne({'_id': document["_id"]}, {
                    '$set': {'lang': 'en'}})
            )
        # Send once every 1000 in batch
        if len(operations) == 1000:
            dbo[subset].bulk_write(operations, ordered=False)
            operations = []

        # if doclang[0] == 'en' and doclang[1] >= 0.70:
        #     print document['text']

    if len(operations) > 0:
        dbo[subset].bulk_write(operations, ordered=False)

    print Counter(lang_dist)
<<<<<<< 6211ca6065b553473e535aa014e2de9685b01659
<<<<<<< 0c748ea8067cc2f81781a61664e964c735fd7a4a
=======
>>>>>>> Keyword search works
    print reduce(lambda x, y: x + y, accuracy) / len(accuracy)


@fileops.do_cprofile
<<<<<<< 6211ca6065b553473e535aa014e2de9685b01659
def keyword_search(client, db_name, keywords):
=======
def keyword_search(client, db_name, lang_list, keywords):
>>>>>>> Keyword search works
    """Perform a text search with the provided keywords in batches of 10.
    Outputs value to new collection

    Args:
        client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        db_name     (str):  Name of database to query.
        lang_list   (list): List of languages to match on.
        keywords    (list): List of keywords to search for.
    """

    # Split the incoming list of keywords into lists of size 10
<<<<<<< 6211ca6065b553473e535aa014e2de9685b01659
    # decomposed_keywords = [keywords[i:i + 10]
    #                        for i in xrange(0, len(keywords), 10)]

    dbo = client[db_name]
    operations = []
    for search_query in keywords:
        # search_query = ' '.join(item)
        print search_query
        pipeline = [
            {"$match": {"$text": {"$search": search_query}}},
            {"$project": {"_id": 1, 'text': 1, 'id': 1, 'timestamp': 1, 'retweeted': 1,
                          'lang': 1, 'user.id_str': 1, 'user.screen_name': 1, 'user.location': 1}}
        ]
        cursor = dbo.tweets.aggregate(pipeline, allowDiskUse=True)
        for document in cursor:
            operations.append(InsertOne(document))

         # Send once every 1000 in batch
        if len(operations) == 1000:
            dbo['keywords'].bulk_write(operations, ordered=False)
            operations = []

    if len(operations) > 0:
        dbo['keywords'].bulk_write(operations, ordered=False)
=======
    decomposed_keywords = [keywords[i:i + 3]
                           for i in xrange(0, len(keywords), 3)]

    dbo = client[db_name]
    for item in decomposed_keywords:
        search_query = ' '.join(item)
        pipeline = [
            {"$match": {"$text": {"$search": search_query}}},
            {"$project": {"_id": 1, 'text': 1, 'id': 1, 'timestamp': 1, 'entities': 1, 'retweeted': 1,
                          'coordinates': 1, 'lang': 1, 'user.id': 1, 'user.screen_name': 1, 'user.location': 1}},
            {"$out": "subset_search_test"}
        ]
        dbo.tweets.aggregate(pipeline, allowDiskUse=True)
>>>>>>> Keyword search works
