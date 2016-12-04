# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module houses various helper functions for use throughout the project
"""

import cProfile
from time import time
from collections import OrderedDict
from modules.utils import constants
from modules.utils import fileops


def unicode_to_utf(unicode_list):
    """ Converts a list of strings from unicode to utf8
    Args:
        unicode_list (list): A list of unicode strings.

    Returns:
        list: UTF8 converted list of strings.
    """
    return [x.encode('UTF8') for x in unicode_list]


def timing(func):
    """Decorator for timing run time of a function
    """

    def wrap(*args):
        """Wrapper
        """
        time1 = time()
        ret = func(*args)
        time2 = time()
        print '%s function took %0.3f ms' % (func.func_name, (time2 - time1) * 1000.0)
        return ret

    return wrap


def do_cprofile(func):
    """Decorator for profiling a function
    """

    def profiled_func(*args, **kwargs):
        """Wrapper
        """
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')

    return profiled_func


def count_sightings(json_obj):
    """ Returns a count of the number of sightings per word in corpus
    Args:
        json_obj (json_obj).

    Returns:
        int: The return value.
    """
    try:
        return int(json_obj['number_of_sightings'])
    except KeyError:
        return 0


def build_query_string(query_words):
    """Builds an OR concatenated string for querying the Twitter Search API.
    Args:
        query_words (list): list of words to be concatenated.

    Returns:
        list: List of words concatenated with OR.
    """
    result = ''.join(
        [q + ' OR ' for q in query_words[0:(len(query_words) - 1)]])
    return result + str(query_words[len(query_words) - 1])


def prep_json_entry(entry):
    """Properly format and return a json object
    """
    json_obj = OrderedDict()
    json_obj['vocabulary'] = entry['vocabulary']
    json_obj['variant_of'] = entry['variant_of']
    json_obj['pronunciation'] = entry['pronunciation']
    json_obj['meaning'] = entry['meaning']
    json_obj['language'] = entry['language']
    json_obj['about_ethnicity'] = entry['about_ethnicity']
    json_obj['about_nationality'] = entry['about_nationality']
    json_obj['about_religion'] = entry['about_religion']
    return json_obj



