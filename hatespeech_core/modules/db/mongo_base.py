# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides simple methods that interact with the MongoDB instance.

Handles connection.
Object and collection retrieval.
Simple aggregations.
Removes all entries from a collection not matching a given language.
"""

from pymongo import MongoClient
from pymongo import errors
from pprint import pprint
from bson.objectid import ObjectId
from bson.son import SON
from ..utils import settings
from ..utils import file_ops


def connect(db_url=None):
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    max_sev_sel_delay = 500
    if db_url is None:
        db_url = settings.DB_URL
    try:
        conn = MongoClient(db_url, serverSelectionTimeoutMS=max_sev_sel_delay, socketKeepAlive=True)
        conn.server_info()
        print "Connected to DB at " + db_url + " successfully"
    except errors.ServerSelectionTimeoutError, ex:
        print "Could not connect to MongoDB: %s" % ex
    return conn


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
    """Fetches k objects from the specified collection.

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


def get_object_ids(connection_params, lang_list, output_name):
    """Creates a collection of consisting only of object ids.

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

    file_ops.write_json_file(output_name, settings.DATA_PATH, result)


def filter_by_language(connection_params, lang_list, output_name):
    """Bulk operation to remove tweets by the lang field.

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