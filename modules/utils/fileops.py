# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
This module houses various file operation functions for use throughout the project
"""

import os
import re
import csv
import string
import json
import glob
import cProfile
from time import time
from collections import OrderedDict
import multiprocessing
from modules.utils import constants
from modules.utils import twokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from joblib import Parallel, delayed
from pymongo import InsertOne


def unicode_to_utf(unicode_list):
    """ Converts a list of strings from unicode to utf8
    Args:
        unicode_list (list): A list of unicode strings.

    Returns:
        list: UTF8 converted list of strings.
    """
    return [x.encode("UTF8") for x in unicode_list]


def timing(func):
    """Decorator for timing run time of a function
    """

    def wrap(*args):
        """Wrapper
        """
        time1 = time()
        ret = func(*args)
        time2 = time()
        print "%s function took %0.3f ms" % (func.func_name, (time2 - time1) * 1000.0)
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
            profile.print_stats(sort="time")

    return profiled_func


def count_sightings(json_obj):
    """ Returns a count of the number of sightings per word in corpus
    Args:
        json_obj (json_obj).

    Returns:
        int: The return value.
    """
    try:
        return int(json_obj["number_of_sightings"])
    except KeyError:
        return 0


def build_query_string(query_words):
    """Builds an OR concatenated string for querying the Twitter Search API.
    Args:
        query_words (list): list of words to be concatenated.

    Returns:
        list: List of words concatenated with OR.
    """
    result = "".join(
        [q + " OR " for q in query_words[0:(len(query_words) - 1)]])
    return result + str(query_words[len(query_words) - 1])


def prep_json_entry(entry):
    """Properly format and return a json object
    """
    json_obj = OrderedDict()
    json_obj["vocabulary"] = entry["vocabulary"]
    json_obj["variant_of"] = entry["variant_of"]
    json_obj["pronunciation"] = entry["pronunciation"]
    json_obj["meaning"] = entry["meaning"]
    json_obj["language"] = entry["language"]
    json_obj["about_ethnicity"] = entry["about_ethnicity"]
    json_obj["about_nationality"] = entry["about_nationality"]
    json_obj["about_religion"] = entry["about_religion"]
    return json_obj


def get_filenames(directory):
    """Reads all the json files in the folder and removes the drive and path and
    extension, only returning a list of strings with the file names.

    Returns:
        list: List of plain filenames
    """
    file_path = glob.glob(directory + "*.json")
    result = []

    for entry in file_path:
        drive, path = os.path.splitdrive(entry)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


def read_json_file(filename, path):
    """Accepts a file name and loads it as a json object.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        obj: json object
    """

    result = []
    try:
        with open(path + filename + ".json", "r") as entry:
            result = json.load(entry)
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_json_file(filename, path, result):
    """Writes the result to json with the given filename.
    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    with open(path + filename + ".json", "w+") as json_file:
        json.dump(result, json_file)
    json_file.close()


def read_csv_file(filename, path):
    """Accepts a file name and loads it as a list.
    Args:
        filename   (str): Filename to be loaded.
        path       (str): Directory path to use.

    Returns:
        list: List of strings.
    """

    try:
        with open(path + filename + ".csv", "r") as entry:
            reader = csv.reader(entry)
            temp = list(reader)
            # flatten to 1D, it gets loaded as 2D array
            result = [x for sublist in temp for x in sublist]
    except IOError as ex:
        print "I/O error({0}): {1}".format(ex.errno, ex.strerror)
    else:
        entry.close()
        return result


def write_csv_file(filename, path, result):
    """Writes the result to csv with the given filename.
    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    output = open(path + filename + ".csv", "wb")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL, lineterminator="\n")
    for val in result:
        writer.writerow([val])
    # Print one a single row
    # writer.writerow(result)


def count_entries(file_list):
    """Performs a count of the number of number of words in the corpus
     Args:
        file_list  (list): list of file names.

    Returns:
        list: A list of json objects containing the count per file name
    """
    result = []
    for obj in file_list:
        with open(constants.CSV_PATH + obj + ".csv", "r") as entry:
            reader = csv.reader(entry, delimiter=",")
            col_count = len(reader.next())
            res = {"Filename": obj, "Count": col_count}
            result.append(res)
    return result


def extract_corpus(file_list):
    """ Loads a set of json files and builds a corpus from the terms within
    Args:
        file_list  (list): list of file names.
    """

    for file_name in file_list:
        json_data = read_json_file(file_name, constants.JSON_PATH)
        result = []

        data = json_data["data"]["datapoint"]
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            result.append(str(entry["vocabulary"]))
        write_csv_file(file_name, constants.CSV_PATH, result)


def json_field_filter(json_obj, field_filter):
    """Accepts a json object and returns only the passed field
     Args:
        json_obj        (obj): json object.
        field_filter    (str): field to extract.

   Returns:
        list: A list of filtered values
    """
    result = []
    for document in json_obj:
        result.append(document[field_filter])
    return result


def filter_hatebase_categories():
    """Filters the hatebase data into categories for black, muslim and latino keywords
    """
    filter1_subset = []
    filter2_subset = []
    filter3_subset = []

    seen_set = set()

    filter1 = ["Black", "black", "Blacks", "blacks",
               "African", "african", "Africans", "africans"]
    filter2 = ["Muslims", "Muslim", "Middle", "Arab", "Arabs", "Arabic"]
    filter3 = ["Hispanic", "hispanic", "Hispanics", "Mexican", "Mexicans", "Latino", "Latinos",
               "Cuban", "Cubans"]

    file_list = get_filenames(constants.JSON_PATH)
    for entry in file_list:
        json_data = read_json_file(entry, constants.JSON_PATH)

        data = json_data["data"]["datapoint"]
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            if any(x in entry["meaning"] for x in filter1):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter1_subset.append(prep_json_entry(entry))

            if any(x in entry["meaning"] for x in filter2):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter2_subset.append(prep_json_entry(entry))

            if any(x in entry["meaning"] for x in filter3):
                if entry["vocabulary"] not in seen_set:
                    seen_set.add(entry["vocabulary"])
                    filter3_subset.append(prep_json_entry(entry))

    write_json_file(
        "filter1_subset", constants.DATA_PATH, filter1_subset)
    write_json_file(
        "filter2_subset", constants.DATA_PATH, filter2_subset)
    write_json_file(
        "filter3_subset", constants.DATA_PATH, filter3_subset)


def parse_category_files():
    """Reads the category entries and return the keywords only

    Returns:
        list: A list of filtered keywords
    """
    result = []
    filter1 = json_field_filter(read_json_file(
        "filter1_subset", constants.DATA_PATH), "vocabulary")
    filter2 = json_field_filter(read_json_file(
        "filter2_subset", constants.DATA_PATH), "vocabulary")
    filter3 = json_field_filter(read_json_file(
        "filter3_subset", constants.DATA_PATH), "vocabulary")

    result = filter1 + filter2 + filter3
    return result


def preprocess_text(raw_text):
    """Preprocessing pipeline for raw text.
    Tokenize, lemmatize and remove stopwords.
    Args:
        raw_text  (str): String to preprocess.

    Returns:
        list: vectorized tweet.
    """
    punctuation = list(string.punctuation)
    stop_list = stopwords.words("english") + punctuation + ["rt", "via", "RT"]

    # Remove urls
    clean_text = re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", raw_text)
    clean_text = twokenize.tokenize(clean_text)

    # Remove numbers
    clean_text = [token for token in clean_text if len(
        token.strip(string.digits)) == len(token)]

    # Record single instances of a term only
    terms_single = set(clean_text)

    # unigrams = [ w for doc in documents for w in doc if len(w)==1]
    # bigrams  = [ w for doc in documents for w in doc if len(w)==2]

    hashtags_only = []
    user_mentions_only = []
    terms_only = []

    for token in terms_single:
        if token.startswith("#"):
            hashtags_only.append(token)

        if token.startswith("@"):
            user_mentions_only.append(token)

        if not token.startswith(("#", "@")) and token not in stop_list:
            terms_only.append(token)

    sentiment = TextBlob(raw_text).sentiment
    return {"hashtags": hashtags_only, "user_mentions": user_mentions_only,
            "tokens": terms_only, "sentiment": list(sentiment)}


def preprocess_tweet(tweet_obj):
    """Preprocessing pipeline for Tweet body.
    Tokenize, lemmatize and remove stopwords.
    Args:
        tweet_obj  (dict): Tweet to preprocess.

    Returns:
        dict: Tweet with vectorized text appended.
    """
    punctuation = list(string.punctuation)
    stop_list = stopwords.words("english") + punctuation + ["rt", "via", "RT"]

    # Remove urls
    clean_text = clean_text = re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", tweet_obj["text"])
    clean_text = twokenize.tokenize(clean_text)

    # Remove numbers
    clean_text = [token for token in clean_text if len(
        token.strip(string.digits)) == len(token)]

    # Record single instances of a term only
    terms_single = set(clean_text)

    # unigrams = [ w for doc in documents for w in doc if len(w)==1]
    # bigrams  = [ w for doc in documents for w in doc if len(w)==2]

    hashtags_only = []
    user_mentions_only = []
    terms_only = []

    for token in terms_single:
        if token.startswith("#"):
            hashtags_only.append(token)

        if token.startswith("@"):
            user_mentions_only.append(token)

        if not token.startswith(("#", "@")) and token not in stop_list:
            terms_only.append(token)

    sentiment = TextBlob(str(terms_single)).sentiment
    tweet_obj["vector"] = {"hashtags": hashtags_only, "user_mentions": user_mentions_only,
                           "tokens": terms_only, "sentiment": list(sentiment)}

    return InsertOne(tweet_obj)


def parallel_preprocess(tweet_list):
    """Passes the incoming raw tweets to our preprocessing function.
    Args:
        tweet_list (list): List of raw tweet texts to preprocess.

    Returns:
        list: List of vectorized tweets.
    """
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(preprocess_tweet)(tweet) for tweet in tweet_list)
    return results
