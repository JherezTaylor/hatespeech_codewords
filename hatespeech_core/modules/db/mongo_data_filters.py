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

from pymongo import InsertOne, DeleteOne, ReplaceOne, UpdateMany


def field_removal(connection_params, lang_list, output_name):
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

    pipeline = [
        UpdateMany({},
                   {
                       "$unset": {
                           "contributors": "", "truncated": "", "retweet_count": "",
                           "retweeted": "", "display_text_range": "", "retweeted_status": "",
                           "extended_entities": "", "entities": "", "favorited": "", "id": "",
                           "user.follow_request_sent": "", "user.profile_use_background_image": "",
                           "user.default_profile_image": "", "user.profile_sidebar_fill_color": "",
                           "user.profile_image_url_https": "", "in_reply_to_user_id": "",
                           "user.profile_text_color": "", "user.profile_sidebar_border_color": "",
                           "user.profile_background_color": "", "in_reply_to_status_id": "",
                           "user.profile_link_color": "", "user.profile_image_url": "",
                           "user.profile_background_tile": "", "extended_tweet": "",
                           "user.notifications": "", "user.default_profile": "",
                           "user.is_translator": "", "user.id": "",
                           "quoted_status": "",
                           "user.profile_background_image_url": "",
                           "user.profile_background_image_url_https": "",

                       },
                       "$set": {"preprocessed": True}}, upsert=False)
    ]
    result = dbo[collection].bulk_write(pipeline, ordered=False)
    return result
