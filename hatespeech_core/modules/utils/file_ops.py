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
from time import time
from collections import OrderedDict
import multiprocessing
import glob
import cProfile
import requests
import ujson
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# from nltk.sentiment.util import demo_sent_subjectivity as SUBJ
from textblob import TextBlob
from joblib import Parallel, delayed
from . import settings
from . import twokenize
# import sys
# reload(sys)
# sys.setdefaultencoding("utf8")


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


@timing
def send_job_notification(title, body):
    """ Send a notification via Pushbullet.

     Args:
        json_obj (json_obj).

    Indicates whether a job has completed or whether an error occured.
    """
    headers = {"Access-Token": settings.PUSHBULLET_API_KEY,
               "Content-Type": "application/json"}
    payload = {"type": "note", "title": title, "body": ujson.dumps(body)}
    url = "https://api.pushbullet.com/v2/pushes"
    return requests.post(url, headers=headers, data=ujson.dumps(payload))


def count_sightings(json_obj):
    """ Returns a count of the number of sightings per word in corpus

    Args:
        json_obj (dict).

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


@do_cprofile
def get_filenames(directory):
    """Reads all the json files names in the directory.

    Returns:
        list: List of plain filenames
    """
    file_path = glob.glob(directory + "*.json")
    result = []

    for entry in file_path:
        _, path = os.path.splitdrive(entry)
        path, filename = os.path.split(path)
        name = os.path.splitext(filename)[0]
        result.append(str(name))
    return result


@do_cprofile
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
            result = ujson.load(entry)
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
        ujson.dump(result, json_file)
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
    """Performs a count of the number of number of words in the corpus.

    Args:
        file_list  (list): list of file names.

    Returns:
        list: A list of json objects containing the count per file name
    """

    result = []
    for obj in file_list:
        with open(settings.CSV_PATH + obj + ".csv", "r") as entry:
            reader = csv.reader(entry, delimiter=",")
            col_count = len(reader.next())
            res = {"Filename": obj, "Count": col_count}
            result.append(res)
    return result


def extract_corpus(file_list):
    """ Loads a set of json files and builds a corpus from the terms within.

    Args:
        file_list  (list): list of file names.
    """

    for file_name in file_list:
        json_data = read_json_file(file_name, settings.JSON_PATH)
        result = []

        data = json_data["data"]["datapoint"]
        data.sort(key=count_sightings, reverse=True)

        for entry in data:
            result.append(str(entry["vocabulary"]))
        write_csv_file(file_name, settings.CSV_PATH, result)


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
    """Filters the hatebase data into categories.

    Does manually parsing for black, muslim and latino keywords.
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

    file_list = get_filenames(settings.JSON_PATH)
    for entry in file_list:
        json_data = read_json_file(entry, settings.JSON_PATH)

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
        "filter1_subset", settings.DATA_PATH, filter1_subset)
    write_json_file(
        "filter2_subset", settings.DATA_PATH, filter2_subset)
    write_json_file(
        "filter3_subset", settings.DATA_PATH, filter3_subset)


def parse_category_files():
    """Reads the category entries and return the keywords only

    Returns:
        list: A list of filtered keywords
    """
    result = []
    filter1 = json_field_filter(read_json_file(
        "filter1_subset", settings.DATA_PATH), "vocabulary")
    filter2 = json_field_filter(read_json_file(
        "filter2_subset", settings.DATA_PATH), "vocabulary")
    filter3 = json_field_filter(read_json_file(
        "filter3_subset", settings.DATA_PATH), "vocabulary")

    result = filter1 + filter2 + filter3
    return result


@timing
def preprocess_text(raw_text):
    """Preprocessing pipeline for raw text.

    Tokenize and remove stopwords.

    Args:
        raw_text  (str): String to preprocess.

    Returns:
        list: vectorized tweet.
    """

    punctuation = list(string.punctuation)
    stop_list = dict.fromkeys(stopwords.words(
        "english") + punctuation + ["rt", "via", "RT"])

    # Remove urls
    clean_text = remove_urls(raw_text)
    clean_text = twokenize.tokenizeRawTweetText(clean_text)

    # Remove numbers
    clean_text = [token for token in clean_text if len(
        token.strip(string.digits)) == len(token)]

    # Record single instances of a term only
    terms_single = set(clean_text)
    terms_single = dict.fromkeys(terms_single)

    # Store any emoji in the text and prevent it from being lowercased then
    # append items marked as Protected by the twokenize library.
    # This protected object covers tokens that shouldn't be split or lowercased so
    # we take advantage of it and use it as quick and dirty way to
    # identify emoji rather than calling a separate regex function.
    emoji = []
    for token in terms_single:
        for match in twokenize.Protected.finditer(token):
            emoji.append(token)

    # If the token is not in the emoji dict, lowercase it
    emoji = dict.fromkeys(emoji)
    for token in terms_single.copy():
        if not token in emoji:
            terms_single[token.lower()] = terms_single.pop(token)

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


def preprocess_tweet(tweet_obj, tweet_split, hs_keywords):
    """Preprocessing pipeline for Tweet body.

    Tokenize and remove stopwords.

    Args:
        tweet_obj   (dict): Tweet to preprocess.
        tweet_split (list): List of tokens in the tweet text.
        hs_keywords (dict): Keywords to match on.

    Returns:
        dict: Tweet with vectorized text appended.
    """

    # Use VADER approach. The compound score is a normalized value of the
    # pos, neg and neu values with -1 being most negative and +1 being most positive.
    # pos, neu and neg are ratios for the proportion of text that fall into
    # each category
    sent_analyzer = SIA()
    sentiment = sent_analyzer.polarity_scores(tweet_obj["text"])

    # Use TextBlob subjectivity for now
    # SUBJ is a trained classifer on the movie dataset in NLTK
    # SUBJ(tweet_obj["text"])
    subjectivity = TextBlob(tweet_obj["text"]).subjectivity

    # Value between -1 and 1 - TextBlob Polarity explanation in layman's
    # terms: http://planspace.org/20150607-textblob_sentiment/

    # Negative sentiment and possibly subjective
    if sentiment["compound"] < 0 and sentiment["neg"] >= 0.5 and subjectivity >= 0.4:
        punctuation = list(string.punctuation)
        stop_list = dict.fromkeys(stopwords.words(
            "english") + punctuation + ["rt", "via", "RT"])

        # Remove urls
        clean_text = clean_text = remove_urls(tweet_obj["text"])
        clean_text = twokenize.tokenizeRawTweetText(clean_text)

        # Remove numbers
        clean_text = [token for token in clean_text if len(
            token.strip(string.digits)) == len(token)]

        # Record single instances of a term only
        terms_single = set(clean_text)
        terms_single = dict.fromkeys(terms_single)

        terms_only = []
        stopwords_only = []

        # Store any emoji in the text and prevent it from being lowercased then
        # append items marked as Protected by the twokenize library.
        # This protected object covers tokens that shouldn't be split or lowercased so
        # we take advantage of it and use it as quick and dirty way to
        # identify emoji rather than calling a separate regex function.
        emoji = []
        for token in terms_single:
            for _match in twokenize.Protected.finditer(token):
                emoji.append(token)

        # If the token is not in the emoji dict, lowercase it
        emoji = dict.fromkeys(emoji)
        for token in terms_single.copy():
            if not token in emoji:
                terms_single[token.lower()] = terms_single.pop(token)

        stopwords_only = set(terms_single).intersection(set(stop_list))
        for token in terms_single:
            if not token.startswith(("#", "@")) and token not in stop_list:
                terms_only.append(token)

        tweet_obj["stopwords"] = list(stopwords_only)
        tweet_obj["tokens"] = terms_only
        tweet_obj["subjectivity"] = subjectivity
        tweet_obj["compound_score"] = sentiment["compound"]
        tweet_obj["hs_keword_count"] = len(
            set(hs_keywords).intersection(tweet_split))
        tweet_obj["neg_score"] = sentiment["neg"]
        tweet_obj["neu_score"] = sentiment["neu"]
        tweet_obj["pos_score"] = sentiment["pos"]
        return tweet_obj
    else:
        pass


def is_garbage(raw_text, precision):
    """ Check if a tweet consists primarly of hashtags, mentions or urls

    Args:
        tweet_obj  (dict): Tweet to preprocess.
    """

    garbage_check = ""
    word_list = raw_text.split()
    for token in word_list:
        if not token.startswith(("#", "@", "http")):
            garbage_check = garbage_check + token + " "

    if ((len(raw_text) - len(garbage_check)) / len(raw_text)) >= precision:
        return True
    else:
        return False


def remove_urls(raw_text):
    """ Removes urls from text

    Args:
        raw_text (str): Text to filter.
    """
    return re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", raw_text)


def parallel_preprocess(tweet_list, tweet_split, hs_keywords):
    """Passes the incoming raw tweets to our preprocessing function.

    Args:
        tweet_list  (list): List of raw tweet texts to preprocess.
        tweet_split (list): List of tokens in the tweet text.
        hs_keywords (dict): Keywords to match on.

    Returns:
        list: List of vectorized tweets.
    """
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(preprocess_tweet)(tweet, tweet_split, hs_keywords) for tweet in tweet_list)
    return results


def create_ngrams(text, length):
    """Create ngrams of the specified length from a string of text
    Args:
        text   (str): Text to process.
        length (int): Length of ngrams to create.
    """

    ngrams = TextBlob(text).ngrams(n=length)
    for index, value in enumerate(ngrams):
        gram = ""
        for pos, token in enumerate(value):
            if pos == len(value) - 1:
                gram += token
            else:
                gram += token + " "
        ngrams[index] = gram
    return ngrams
