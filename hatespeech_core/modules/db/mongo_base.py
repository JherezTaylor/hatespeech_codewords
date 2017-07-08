# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module provides simple methods that interact with the MongoDB instance.

Handles connection.
Object and collection retrieval.
Simple aggregations.
Removes all entries from a collection not matching a given language.
"""

from time import sleep
from pymongo import MongoClient, UpdateOne, errors
from bson.objectid import ObjectId
from bson.son import SON
from ..utils import settings
from ..utils import file_ops
from ..utils import twitter_api


def connect(db_url=None):
    """Initializes a pymongo conection object.

    Returns:
        pymongo.MongoClient: Connection object for Mongo DB_URL
    """
    max_sev_sel_delay = 7000
    if db_url is None:
        db_url = settings.DB_URL
    try:
        conn = MongoClient(
            db_url, serverSelectionTimeoutMS=max_sev_sel_delay, socketKeepAlive=True)
        conn.server_info()
        settings.logger.info("Connected to DB at %s successfully", db_url)
        return conn
    except errors.ServerSelectionTimeoutError as ex:
        settings.logger.error(
            "Could not connect to MongoDB: %s", ex, exc_info=True)
        raise


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
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
    cursor = dbo[collection].find({"_id": ObjectId(object_id)})
    return list(cursor)


def finder(connection_params, query, count):
    """Fetches k objects from the specified collection.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
        query  (dict): Query to execute.
        count: (bool): Execute a count query.
    Returns: Pymongo cursor or count
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
    if not any(query["projection"]):
        query["projection"] = None

    if count:
        cursor = dbo[collection].find(filter=query["filter"], projection=query[
            "projection"], skip=query["skip"], limit=query["limit"]).count()
        return cursor
    else:
        cursor = dbo[collection].find(filter=query["filter"], projection=query["projection"], skip=query[
                                      "skip"], limit=query["limit"], no_cursor_timeout=query["no_cursor_timeout"])
        return cursor


def aggregate(connection_params, pipeline):
    """ Executes the specified aggregation pipeline.
    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        query  (list): List storing aggregation dict params.
    """

    client = connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
    return dbo[collection].aggregate(pipeline, allowDiskUse=True)


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
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
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
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

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
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
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
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
    pipeline = [
        {"$match": {"lang": {"$in": lang_list}}},
        {"$project": {"_id": 1}},
        {"$out": output_name}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)
    cursor = dbo[output_name].find({})

    result = [str(document["_id"]) for document in cursor]
    file_ops.write_json_file(output_name, settings.OUTPUT_PATH, result)


def update_missing_text(connection_params):
    """ Some tweets are missing the text field, possibly an oversight
    during cleanup. Retrieve matching tweets, call the Twitter API and update the text field.

    Args:
    connection_params   (list): Contains connection objects and params as follows:
        0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
        1: db_name     (str): Name of database to query.
        2: collection  (str): Name of collection to use.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)
    cursor = dbo[collection].find(
        {"text": None}, {"id_str": 1}, no_cursor_timeout=True)

    operations = []
    raw_docs = {}

    request_count = 0
    found_count = 0
    not_found_count = 0

    for document in cursor:
        raw_docs[document["id_str"]] = document["_id"]

        if (len(raw_docs) % 100) == 0:
            request_count += 1

            response = do_missing_text_op(raw_docs, request_count)
            settings.logger.debug("Req Count: %s", request_count)
            operations += response[0]
            found_count += response[1]
            not_found_count += response[2]
            raw_docs.clear()

        if len(operations) == settings.BULK_BATCH_SIZE:
            _ = do_bulk_op(dbo, collection, operations)
            operations = []

    # Exit loop and flush remaining entries
    if (len(raw_docs) % 100) != 0:
        request_count += 1
        settings.logger.debug("Req Count: %s", request_count)
        response = do_missing_text_op(raw_docs, request_count)
        operations += response[0]
        found_count += response[1]
        not_found_count += response[2]
        raw_docs.clear()

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = do_bulk_op(dbo, collection, operations)
    # Do a manual delete many to clean up tweets that weren't found
    return [found_count, not_found_count, request_count]


def do_missing_text_op(raw_docs, request_count):
    """ Apply the insert operation.

    Args:
        raw_docs        (dict): Contains id_str(k) and _id(v) as key value pairs.
        request_count   (int): Number of requests sent to Twitter API so far.
    """

    found_count = 0
    operations = []

    # Obey Twitter Rate Limit
    if request_count != 0 and (request_count % 900) == 0:
        sleep(60 * 15)
    else:
        pass

    api_response = twitter_api.run_status_lookup(list(raw_docs.keys()))
    for doc in api_response:
        if doc["id_str"] in raw_docs:
            found_count += 1
            operations.append(UpdateOne({"_id": raw_docs[doc["id_str"]]}, {
                "$set": {"text": doc["text"]}}, upsert=False))
        else:
            pass
    not_found_count = len(raw_docs) - len(api_response)
    return [operations, found_count, not_found_count]


def do_bulk_op(dbo, collection, operations):
    """ Execute bulk operations and return the result.

    Args:
        dbo (Pymongo DB): DB object
        collection (str): Name of collection to use.
        operations (list): List of MongoDB operations.
    """

    try:
        result = dbo[collection].bulk_write(operations, ordered=False)
        return result
    except errors.BulkWriteError as bwe:
        settings.logger.error(bwe.details, exc_info=True)
        settings.logger.error(len(operations))
        settings.logger.error(operations)
        raise
