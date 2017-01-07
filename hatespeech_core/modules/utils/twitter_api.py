# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module provides functions that request data from the Twitter API through Twython
"""

from twython import Twython, TwythonRateLimitError, TwythonError
from . import settings


def run_status_lookup(id_list):
    """Make a call to statuses/lookup
     Args:
       id_list   (list): Contains the tweet ids to  lookup
    """
    twitter = Twython(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET,
                      settings.TWITTER_OAUTH, settings.TWITTER_OAUTH_SECRET)

    query_string = ",".join(id_list)
    try:
        result = twitter.lookup_status(
            id=query_string, include_entities=False, trim_user=True)
    except TwythonRateLimitError:
        print "Rate limit reached, taking a break for a minute...\n"
    except TwythonError as err:
        print "Some other error occured, taking a break for half a minute: " + str(err)
    return result
