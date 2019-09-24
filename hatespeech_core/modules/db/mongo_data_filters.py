# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module provides methods that focus on removing and compressing fields.

Aggreagate users and hashtags.
Map reduce samples.
Keyword searches. Only provided as a sample, for large collections ElasticSearch should be used.
Identify the language of tweets with an und lang.
"""

from bs4 import BeautifulSoup as BSHTML
from pymongo import UpdateOne, DeleteMany, UpdateMany, ASCENDING
from ..utils import settings
from ..utils import notifiers
from . import mongo_base


def retweet_removal(connection_params):
    """Bulk operation to delete all retweets.

    Preprocessing Pipeline Stage 1.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    pipeline = [
        DeleteMany({"retweeted_status": {"$exists": True}})
    ]

    result = mongo_base.do_bulk_op(dbo, collection, pipeline)
    return result


def create_indexes(connection_params):
    """Create index for field existence checks

    Preprocessing Pipeline Stage 2.

    Filters by any lang not in lang_list. This should ideally be
    run directly through mongo shellfor large collections.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    dbo[collection].create_index(
        [("entities.user_mentions", ASCENDING)], background=True)
    settings.logger.info("User mentions Index built")

    dbo[collection].create_index(
        [("entities.hashtags", ASCENDING)], background=True)
    settings.logger.info("Hashtag Index built")

    dbo[collection].create_index(
        [("quoted_status_id", ASCENDING)], background=True)
    settings.logger.info("Quoted status Index built")

    dbo[collection].create_index(
        [("extended_tweet.id_str", ASCENDING)], background=True)
    settings.logger.info("Extended tweet Index built")

    dbo[collection].create_index(
        [("quoted_status.entities.hashtags", ASCENDING)], background=True)
    settings.logger.info("Quoted status hashtag Index built")

    dbo[collection].create_index(
        [("quoted_status.entities.user_mentions", ASCENDING)], background=True)
    settings.logger.info("Quoted status user_mention Index built")


@notifiers.timing
def field_removal(connection_params):
    """Bulk operation to remove unwanted fields from the tweet object

    Preprocessing Pipeline Stage 3.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

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

    result = mongo_base.do_bulk_op(dbo, collection, pipeline)
    return result


@notifiers.timing
def quoted_status_field_removal(connection_params):
    """Bulk operation to remove unwanted fields from the quoted_status tweet object

    Preprocessing Pipeline Stage 3.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

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

    result = mongo_base.do_bulk_op(dbo, collection, pipeline)
    return result


@notifiers.timing
def language_trimming(connection_params, lang_list):
    """Bulk operation to trim the list of languages present.

    Preprocessing Pipeline Stage 4.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        lang_list (list): List of languages to match on.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    pipeline = [
        DeleteMany({"lang": {"$nin": lang_list}})
    ]

    result = mongo_base.do_bulk_op(dbo, collection, pipeline)
    return result


@notifiers.do_cprofile
def field_flattening_base(connection_params, depth, field_name, field_to_set, field_to_extract):
    """Aggregate operation to unwind entries in the various entities object.

    Preprocessing Pipeline Stage 5.
    Entities include hashtags, user_mentions, urls and media.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        depth (str): Extract from top level of tweet or from nested quote tweet.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    if depth == "top_level":
        field_name_base = field_name
        field_name = "$" + field_name
    elif depth == "quoted_status":
        field_name_base = "quoted_status." + field_name
        field_name = "$" + "quoted_status." + field_name
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

        # Send once every settings.BULK_BATCH_SIZE in batch
        if (len(operations) % settings.BULK_BATCH_SIZE) == 0:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)

    # Clean Up
    dbo["temp_" + field_name_base].drop()


def field_flattening_complex(connection_params, depth, field_params):
    """Aggregate operation to unwind entries in the various entities object.

    Preprocessing Pipeline Stage 5.
    Entities include hashtags, user_mentions, urls and media.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        depth (str): Extract from top level of tweet or from nested quote tweet.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    field_name = field_params[0]
    field_to_set_1 = field_params[2]
    field_to_set_2 = field_params[3]
    field_to_extract_1 = field_params[4]
    field_to_extract_2 = field_params[5]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    if depth == "top_level":
        field_name_base = field_name
        field_name = "$" + field_name
        field_top_level = field_params[1]
        insertion_field = field_top_level

    elif depth == "quoted_status":
        field_top_level = field_params[1]
        field_name_base = "quoted_status." + field_name
        field_name = "$" + "quoted_status." + field_name
        insertion_field = "quoted_status." + field_top_level
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
                              insertion_field: document[field_top_level],
                              str(insertion_field) + "_extracted": True
                          },
                          "$unset": {
                              str(field_name_base): ""
                          }
            }, upsert=False))

        # Send once every settings.BULK_BATCH_SIZE in batch
        if (len(operations) % settings.BULK_BATCH_SIZE) == 0:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)

    # Clean Up
    dbo["temp_" + field_name_base].drop()


def parse_extended_tweet(connection_params, depth):
    """Aggregate operation to parse extended tweet and append contents to top level.

    Preprocessing Pipeline Stage 6.
    Entities include hashtags, user_mentions, urls and media.
    http://stackoverflow.com/questions/28827516/using-unwind-twice-with-group-and-sum-mongodb

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        depth (str): Extract from top level of tweet or from nested quote tweet.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    hashtag_field = "hashtags"
    user_mention_field = "user_mentions"
    url_field = "urls"
    media_field = "media"

    if depth == "top_level":
        field_name_base = "extended_tweet"
        field_name = "$" + "extended_tweet"

    elif depth == "quoted_status":
        field_name_base = "quoted_status.extended_tweet"
        field_name = "$" + "quoted_status.extended_tweet"

    pipeline_hashtags = [
        {"$match": {field_name_base + ".entities.hashtags": {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name + ".entities.hashtags"},
        {"$group": {
            "_id": "$_id",
            "full_text": {"$first": field_name + ".full_text"},
            "hashtags": {"$addToSet": field_name + ".entities.hashtags.text"}
        }},
        {"$out": "temp_hs_" + field_name_base}
    ]

    pipeline_user_mentions = [
        {"$match": {field_name_base + ".entities.user_mentions": {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name + ".entities.user_mentions"},
        {"$group": {
            "_id": "$_id",
            "full_text": {"$first": "$full_text"},
            "user_mentions": {"$addToSet": {"screen_name": field_name +
                                            ".entities.user_mentions.screen_name",
                                            "id_str": field_name + ".entities.user_mentions.id_str"}},
        }},
        {"$out": "temp_um" + field_name_base}
    ]

    pipeline_urls = [
        {"$match": {field_name_base + ".entities.urls": {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name + ".entities.urls"},
        {"$group": {
            "_id": "$_id",
            "full_text": {"$first": "$full_text"},
            "urls": {"$addToSet": field_name + ".entities.urls.expanded_url"}
        }},
        {"$out": "temp_url" + field_name_base}
    ]

    pipeline_media = [
        {"$match": {field_name_base + ".entities.media": {"$exists": True}}},
        {"$project": {field_name_base: 1, "_id": 1}},
        {"$unwind": field_name + ".entities.media"},
        {"$group": {
            "_id": "$_id",
            "full_text": {"$first": "$full_text"},
            "media": {"$addToSet": {"media_url": field_name + ".entities.media.media_url",
                                    "id_str": field_name + ".entities.media.type"}},
        }},
        {"$out": "temp_md" + field_name_base}
    ]

    dbo[collection].aggregate(pipeline_hashtags, allowDiskUse=True)
    iterate_cursor(dbo, "temp_hs_" + field_name_base,
                   collection, hashtag_field, depth)

    dbo[collection].aggregate(pipeline_user_mentions, allowDiskUse=True)
    iterate_cursor(dbo, "temp_um" + field_name_base,
                   collection, user_mention_field, depth)

    dbo[collection].aggregate(pipeline_urls, allowDiskUse=True)
    iterate_cursor(dbo, "temp_url" + field_name_base,
                   collection, url_field, depth)

    dbo[collection].aggregate(pipeline_media, allowDiskUse=True)
    iterate_cursor(dbo, "temp_md" + field_name_base,
                   collection, media_field, depth)


def iterate_cursor(dbo, source_collection, target_collection, field_to_set, depth):
    """ Iterate the specified collections and apply the updates

    Args:
        dbo (MongoClient):    MongoClient connection object
        source_collection  (str):  Collection containing aggregate results.
        target_collection  (str):  Collection to update.
        field_to_set (str):  Name of field to append to collection.
        depth (str): Extract from top level of tweet or from nested quote tweet.
    """

    # Store the documents for our bulkwrite
    if depth == "top_level":
        field = field_to_set
        field_to_set = field_to_set

    elif depth == "quoted_status":
        field = field_to_set
        field_to_set = "quoted_status." + field_to_set
    operations = []
    cursor = dbo[source_collection].find({}, no_cursor_timeout=True)

    for document in cursor:
        # Some tweets appear to be missing text, do this check just in case
        if document["full_text"]:
            operations.append(
                UpdateOne({"_id": document["_id"]},
                          {
                    "$set": {
                        "text": document["full_text"],
                        field_to_set: document[field],
                        "extended_tweet_extracted": True
                    }
                }, upsert=False))
        else:
            operations.append(
                UpdateOne({"_id": document["_id"]},
                          {
                    "$set": {
                        field_to_set: document[field],
                        "extended_tweet_extracted": True
                    }
                }, upsert=False))

        # Send once every settings.BULK_BATCH_SIZE in batch
        if (len(operations) % settings.BULK_BATCH_SIZE) == 0:
            _ = mongo_base.do_bulk_op(dbo, target_collection, operations)
            operations = []

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = mongo_base.do_bulk_op(dbo, target_collection, operations)

    # Clean Up
    dbo[source_collection].drop()


@notifiers.timing
def final_field_removal(connection_params):
    """Bulk operation to remove unwanted fields from the tweet object

    Preprocessing Pipeline Stage 7.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    dbo[collection].drop_indexes()
    pipeline = [
        UpdateMany({},
                   {
                       "$unset": {
                           "entities": "", "quoted_status.entities": "",
                           "id": "", "quoted_status.id": "", "quoted_status_id": "",
                           "quoted_status.quoted_status_id": "", "quoted_status.extended_entities": "",
                           "extended_entities": "", "extended_tweet": "", "quoted_status.extended_tweet": ""
                       }}, upsert=False)
    ]
    result = mongo_base.do_bulk_op(dbo, collection, pipeline)
    return result


def clean_source_field(connection_params):
    """Parse the HTML in the source field.

    Preprocessing Pipeline Stage 8.

    Args:
        connection_params  (list): Contains connection params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
    """

    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]

    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    operations = []

    cursor = dbo[collection].find({"source": {"$exists": True}}, {
                                  "source": 1}, no_cursor_timeout=True)
    for document in cursor:
        try:
            cleaned_source = BSHTML(
                "'" + document["source"] + "'", "html.parser").a.contents[0].encode('utf-8').strip()
        except AttributeError:
            cleaned_source = document["source"]

        operations.append(
            UpdateOne({"_id": document["_id"]},
                      {
                          "$set": {
                              "source": cleaned_source
                          }
            }, upsert=False))

        # Send once every settings.BULK_BATCH_SIZE in batch
        if (len(operations) % settings.BULK_BATCH_SIZE) == 0:
            _ = mongo_base.do_bulk_op(dbo, collection, operations)
            operations = []

    if (len(operations) % settings.BULK_BATCH_SIZE) != 0:
        _ = mongo_base.do_bulk_op(dbo, collection, operations)
