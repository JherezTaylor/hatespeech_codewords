# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides functions that request data from the Twitter API through Twython
"""

from twython import Twython, TwythonRateLimitError, TwythonError
from . import settings
from . import file_ops


def run_status_lookup(id_list):
    """Make a call to statuses/lookup
     Args:
       id_list (list): Contains the tweet ids to lookup
    """

    twitter = Twython(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET,
                      settings.TWITTER_OAUTH, settings.TWITTER_OAUTH_SECRET)

    query_string = ",".join(id_list)
    try:
        result = twitter.lookup_status(
            id=query_string, include_entities=False, trim_user=True)
    except TwythonRateLimitError:
        settings.logger.error(
            "Rate limit reached, taking a break for a minute...\n")
    except TwythonError as err:
        settings.logger.error(
            "Some other error occured, taking a break for half a minute: " + str(err), exc_info=True)
    return result

def lookup_ids(id_list):
    twitter = Twython(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET,
                      settings.TWITTER_OAUTH, settings.TWITTER_OAUTH_SECRET)

    query_string = ",".join(id_list)
    try:
        result = twitter.lookup_user(
            user_id=query_string, include_entities=False)
    except TwythonRateLimitError:
        settings.logger.error(
            "Rate limit reached, taking a break for a minute...\n")
    except TwythonError as err:
        settings.logger.error(
            "Some other error occured, taking a break for half a minute: " + str(err), exc_info=True)
    return result

def check_deleted():
    lookup_list = file_ops.read_csv_file(
        'melvyn_hs_user_ids', settings.TWITTER_SEARCH_PATH)
    batch = set()
    input_count = len(lookup_list)
    request_count = 0
    found_count = 0
    not_found_count = 0
    for document in lookup_list:
        batch.add(document)

        if (len(batch) % 100) == 0:
            request_count += 1
            print(request_count)
            api_response = lookup_ids(list(batch))
            for doc in api_response:
                if doc["id_str"] in batch:
                    found_count += 1
                else:
                    pass
            not_found_count += len(batch) - len(api_response)
            batch = set()
    print("Original num ID: ", input_count)
    print("Found count: ", found_count)
    print("Not found count: ", not_found_count)