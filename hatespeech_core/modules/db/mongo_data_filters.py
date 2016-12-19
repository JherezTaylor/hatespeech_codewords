# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides methods that focus on removing and compressing fields.

Aggreagate users and hashtags.
Map reduce samples.
Keyword searches. Only provided as a sample, for large collections ElasticSearch should be used.
Identify the language of tweets with an und lang.
"""

from pymongo import InsertOne, UpdateOne, DeleteOne, DeleteMany, ReplaceOne, UpdateMany, ASCENDING, errors
from ..utils import file_ops


def retweet_removal(connection_params):
    """Bulk operation to delete all retweets.

    Prerocessing Pipeline Stage 1.

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

    pipeline = [
        DeleteMany({"retweeted_status": {"$exists": True}})
    ]

    result = dbo[collection].bulk_write(pipeline, ordered=False)
    return result


def create_indexes(connection_params):
    """Create index for field existence checks

    Prerocessing Pipeline Stage 2.

    Filters by any lang not in lang_list. This should ideally be
    run directly through mongo shellfor large collections.

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

    dbo[collection].create_index(
        [("entities.user_mentions", ASCENDING)], sparse=True, background=True)
    print "User mentions Index built"

    dbo[collection].create_index(
        [("entities.hashtags", ASCENDING)], sparse=True, background=True)
    print "Hashtag Index built"

    dbo[collection].create_index(
        [("quoted_status_id", ASCENDING)], sparse=True, background=True)
    print "Quoted status Index built"

    dbo[collection].create_index(
        [("extended_tweet.id_str", ASCENDING)], sparse=True, background=True)
    print "Extended tweet Index built"

    dbo[collection].create_index(
        [("quoted_status.entities.hashtags", ASCENDING)], sparse=True, background=True)
    print "Quoted status hashtag Index built"

    dbo[collection].create_index(
        [("quoted_status.entities.user_mentions", ASCENDING)], sparse=True, background=True)
    print "Quoted status user_mention Index built"


@file_ops.timing
def field_removal(connection_params):
    """Bulk operation to remove unwanted fields from the tweet object

    Prerocessing Pipeline Stage 3.

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

    pipeline = [
        UpdateMany({},
                   {
                       "$unset": {
                           "contributors": "", "truncated": "", "retweet_count": "",
                           "retweeted": "", "favorited": "",
                           "user.follow_request_sent": "", "user.profile_use_background_image": "",
                           "user.default_profile_image": "", "user.profile_sidebar_fill_color": "",
                           "user.profile_text_color": "", "user.profile_sidebar_border_color": "",
                           "user.profile_image_url_https": "", "in_reply_to_user_id": "",
                           "user.profile_background_color": "", "in_reply_to_status_id": "",
                           "user.profile_link_color": "", "geo": "",
                           "user.profile_image_url": "", "following": "",
                           "user.profile_background_tile": "", "user.contributors_enabled": "",
                           "user.notifications": "", "user.is_translator": "", "user.id": "",
                           "user.profile_background_image_url": "", "user.has_extended_profile": "",
                           "user.profile_background_image_url_https": "",
                           "user.is_translation_enabled": "", "metadata": "",
                           "user.translator_type": "",

                       },
                       "$set": {"fields_removed": True}}, upsert=False)
    ]

    try:
        result = dbo[collection].bulk_write(pipeline, ordered=False)
    except errors.BulkWriteError as bwe:
        print bwe.details
        werrors = bwe.details['writeErrors']
        print werrors
        raise
    return result


@file_ops.timing
def language_trimming(connection_params, lang_list):
    """Bulk operation to trim the list of languages present.

    Prerocessing Pipeline Stage 4.

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

    pipeline = [
        DeleteMany({"lang": {"$nin": lang_list}})
    ]

    result = dbo[collection].bulk_write(pipeline, ordered=False)
    return result


@file_ops.do_cprofile
def field_flattening_base(connection_params, field_name, field_to_set, field_to_extract):
    """Aggregate operation to unwind entries in the various entities object.

    Prerocessing Pipeline Stage 5.
    Entities include hashtags, user_mentions, urls and media.

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

    field_name_base = field_name
    field_name = "$" + field_name
    # Store the documents for our bulkwrite
    operations = []

    pipeline = [
        {"$match": {field_name_base: {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name},
        {"$group": {"_id": "$_id", field_to_set: {
            "$addToSet": field_name + field_to_extract}}},
        {"$out": "temp_" + field_name_base}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)
    cursor = dbo["temp_" + field_name_base].find({}, no_cursor_timeout=True)

    for document in cursor:
        operations.append(
            UpdateOne({"_id": document["_id"]},
                      {
                          "$set": {
                              field_to_set: document[field_to_set],
                              str(field_to_set) + "_extracted": True
                          },
                          "$unset": {
                              str(field_name_base): ""
                          }
            }, upsert=False))

        # Send once every 1000 in batch
        if (len(operations) % 1000) == 0:
            dbo[collection].bulk_write(operations, ordered=False)
            operations = []

    if (len(operations) % 1000) != 0:
        dbo[collection].bulk_write(operations, ordered=False)

    # Clean Up
    dbo["temp_" + field_name_base].drop()


def field_flattening_complex(connection_params, field_params):
    """Aggregate operation to unwind entries in the various entities object.

    Prerocessing Pipeline Stage 5.
    Entities include hashtags, user_mentions, urls and media.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: client      (pymongo.MongoClient): Connection object for Mongo DB_URL.
            1: db_name     (str): Name of database to query.
            2: collection  (str): Name of collection to use.
    """

    client = connection_params[0]
    db_name = connection_params[1]
    collection = connection_params[2]

    field_name = field_params[0]
    field_top_level = field_params[1]
    field_to_set_1 = field_params[2]
    field_to_set_2 = field_params[3]
    field_to_extract_1 = field_params[4]
    field_to_extract_2 = field_params[5]

    dbo = client[db_name]

    field_name_base = field_name
    field_name = "$" + field_name
    # Store the documents for our bulkwrite
    operations = []

    pipeline = [
        {"$match": {field_name_base: {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name},
        {"$group": {"_id": "$_id", field_top_level:
                    {"$addToSet": {field_to_set_1: field_name + field_to_extract_1,
                                   field_to_set_2: field_name + field_to_extract_2}
                     }
                    }
         },
        {"$out": "temp_" + field_name_base}
    ]

    dbo[collection].aggregate(pipeline, allowDiskUse=True)
    cursor = dbo["temp_" + field_name_base].find({}, no_cursor_timeout=True)

    for document in cursor:
        operations.append(
            UpdateOne({"_id": document["_id"]},
                      {
                          "$set": {
                              field_top_level: document[field_top_level],
                              str(field_top_level) + "_extracted": True
                          },
                          "$unset": {
                              str(field_name_base): ""
                          }
            }, upsert=False))

        # Send once every 1000 in batch
        if (len(operations) % 1000) == 0:
            dbo[collection].bulk_write(operations, ordered=False)
            operations = []

    if (len(operations) % 1000) != 0:
        dbo[collection].bulk_write(operations, ordered=False)

    # Clean Up
    dbo["temp_" + field_name_base].drop()


@file_ops.timing
def quoted_status_field_removal(connection_params):
    """Bulk operation to remove unwanted fields from the quoted_status tweet object

    Prerocessing Pipeline Stage 6.

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

    pipeline = [
        UpdateMany({"quoted_status": {"$exists": True}},
                   {
                       "$unset": {
                           "quoted_status.contributors": "", "quoted_status.truncated": "",
                           "quoted_status.retweeted": "", "quoted_status.favorited": "",
                           "quoted_status.user.follow_request_sent": "", "quoted_status.user.profile_use_background_image": "",
                           "quoted_status.user.default_profile_image": "", "quoted_status.user.profile_sidebar_fill_color": "",
                           "quoted_status.user.profile_text_color": "", "quoted_status.user.profile_sidebar_border_color": "",
                           "quoted_status.user.profile_image_url_https": "", "quoted_status.in_reply_to_user_id": "",
                           "quoted_status.user.profile_background_color": "", "quoted_status.in_reply_to_status_id": "",
                           "quoted_status.user.profile_link_color": "", "quoted_status.geo": "",
                           "quoted_status.user.profile_image_url": "", "quoted_status.following": "",
                           "quoted_status.user.profile_background_tile": "", "quoted_status.user.contributors_enabled": "",
                           "quoted_status.user.notifications": "", "quoted_status.user.is_translator": "", "quoted_status.user.id": "",
                           "quoted_status.user.profile_background_image_url": "", "quoted_status.user.has_extended_profile": "",
                           "quoted_status.user.profile_background_image_url_https": "",
                           "quoted_status.user.is_translation_enabled": "", "quoted_status.metadata": "",
                           "quoted_status.user.translator_type": "",

                       },
                       "$set": {"fields_removed": True}}, upsert=False)
    ]

    try:
        result = dbo[collection].bulk_write(pipeline, ordered=False)
    except errors.BulkWriteError as bwe:
        print bwe.details
        werrors = bwe.details['writeErrors']
        print werrors
        raise
    return result
