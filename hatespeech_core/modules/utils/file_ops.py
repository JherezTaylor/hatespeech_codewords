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
import copy
from collections import OrderedDict
import glob
import cProfile
import pstats
from itertools import chain
import requests
import ujson
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# from nltk.sentiment.util import demo_sent_subjectivity as SUBJ
from textblob import TextBlob
from joblib import Parallel, delayed, cpu_count
from . import settings
from . import twokenize
from . import EmotionDetection

PUNCTUATION = list(string.punctuation)
STOP_LIST = set(stopwords.words(
    "english") + PUNCTUATION + ["rt", "via", "RT"])


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
        print("%s function took %0.3f ms" %
              (func.__name__, (time2 - time1) * 1000.0))
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
            stats = pstats.Stats(profile)
            stats.sort_stats("time").print_stats(20)

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


def send_job_completion(run_time, args):
    """Format and print the details of a completed job

    Args:
        run_time (list): Start and end times.
        args (list): Contains the following:
            0: function_name (str): Name of the function that was run.
            1: message_text  (str): Text to be sent in notification.
    """

    time_diff = round((run_time[1] - run_time[0]), 2)
    print("%s function took %0.3f seconds" % (args[0], time_diff))
    send_notification = send_job_notification(
        settings.MONGO_SOURCE + ": " + args[1] + " took " + str(time_diff) + " seconds", "Complete")
    print(send_notification.content)


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
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
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
        print("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    else:
        entry.close()
        return result


def write_csv_file(filename, path, result):
    """Writes the result to csv with the given filename.

    Args:
        filename   (str): Filename to write to.
        path       (str): Directory path to use.
    """

    output = open(path + filename + ".csv", "w")
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
            col_count = len(next(reader))
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

        result = [str(entry["vocabulary"]) for entry in data]
        write_csv_file(file_name, settings.CSV_PATH, result)


def json_field_filter(json_obj, field_filter):
    """Accepts a json object and returns only the passed field

    Args:
        json_obj        (obj): json object.
        field_filter    (str): field to extract.

    Returns:
        list: A list of filtered values
    """
    result = [document[field_filter] for document in json_obj]
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
        "filter1_subset", settings.OUTPUT_PATH, filter1_subset)
    write_json_file(
        "filter2_subset", settings.OUTPUT_PATH, filter2_subset)
    write_json_file(
        "filter3_subset", settings.OUTPUT_PATH, filter3_subset)


def parse_category_files():
    """Reads the category entries and return the keywords only

    Returns:
        list: A list of filtered keywords
    """
    result = []
    filter1 = json_field_filter(read_json_file(
        "filter1_subset", settings.OUTPUT_PATH), "vocabulary")
    filter2 = json_field_filter(read_json_file(
        "filter2_subset", settings.OUTPUT_PATH), "vocabulary")
    filter3 = json_field_filter(read_json_file(
        "filter3_subset", settings.OUTPUT_PATH), "vocabulary")

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

    cleaned_result = clean_tweet_text(raw_text)
    terms_single = cleaned_result[1]

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

        if not token.startswith(("#", "@")) and token not in STOP_LIST:
            terms_only.append(token)

    sentiment = TextBlob(raw_text).sentiment
    return {"hashtags": hashtags_only, "user_mentions": user_mentions_only,
            "tokens": terms_only, "sentiment": list(sentiment)}


def get_sent_subj(raw_text):
    """ Return the subjectivity and sentiment scores of the text.

    Args:
        raw_text  (str): String to preprocess.

    Returns:
        list: subjectivity and sentiment scores.
    """

    # Use VADER approach. The compound score is a normalized value of the
    # pos, neg and neu values with -1 being most negative and +1 being most positive.
    # pos, neu and neg are ratios for the proportion of text that fall into
    # each category
    sent_analyzer = SIA()
    sentiment = sent_analyzer.polarity_scores(raw_text)

    # Use TextBlob subjectivity for now
    subjectivity = TextBlob(raw_text).subjectivity
    return [sentiment, subjectivity]


def preprocess_tweet(tweet_obj, hs_keywords, args):
    """Preprocessing pipeline for Tweet body.

    Tokenize and remove stopwords.

    Args:
        tweet_obj   (dict): Tweet to preprocess.
        hs_keywords (set): Keywords to match on.
        args        (list): Contains a list of conditions as follows:
            0: subj_check (bool): Check text for subjectivity.
            1: sent_check (bool): Check text for sentiment.
            2: intensity_measure (bool) Retain tokens with all uppercase.

    Returns:
        dict: Tweet with vectorized text appended.
    """

    # SUBJ is a trained classifer on the movie dataset in NLTK
    # SUBJ(tweet_obj["text"])

    subj_check = args[0]
    sent_check = args[1]

    if subj_check and sent_check:
        subj_sent = get_sent_subj(tweet_obj["text"])
        sentiment = subj_sent[0]
        subjectivity = subj_sent[1]
        # Value between -1 and 1 - TextBlob Polarity explanation in layman's
        # terms: http://planspace.org/20150607-textblob_sentiment/

        # Negative sentiment and possibly subjective
        if sentiment["compound"] < 0 and sentiment["neg"] >= 0.5 and subjectivity >= 0.4:

            processed_text = prepare_text(
                tweet_obj["text"].lower(), [STOP_LIST, hs_keywords])

            tweet_obj = prep_tweet_body(
                tweet_obj, [True, subjectivity, sentiment], processed_text)
            return tweet_obj
        else:
            pass

    if subj_check and not sent_check:
        pass

    if not subj_check and sent_check:
        pass

    elif not subj_check and not sent_check:
        processed_text = prepare_text(
            tweet_obj["text"].lower(), [STOP_LIST, hs_keywords])

        tweet_obj = prep_tweet_body(
            tweet_obj, [False], processed_text)
        return tweet_obj


def prepare_text(raw_text, args):
    """ Clean and tokenize text. Create ngrams.
    Ensure that raw_text is lowercased if that's required.

    Args:
        raw_text  (str): String to preprocess.
        args      (list): Contains the following:
            0: stop_list (list): English stopwords and punctuation.
            1: hs_keywords (list) HS corpus.
            2: intensity_measure (bool) Retain tokens with all uppercase.

    Returns:
        list: terms_only, stopword_only, hashtags, mentions, hs_keyword_count, ngrams,
        stopword_ngrams
    """

    stop_list = args[0]
    hs_keywords = args[1]

    cleaned_result = clean_tweet_text(raw_text)
    clean_text = cleaned_result[0]
    terms_single = cleaned_result[1]

    stopwords_only = set(terms_single).intersection(stop_list)
    terms_only = []
    hashtags = []
    mentions = []

    for token in terms_single:
        if not token.startswith(("#", "@")) and token not in stop_list:
            terms_only.append(token)
        if token.startswith(("#")):
            hashtags.append(token)
        if token.startswith(("@")):
            mentions.append(token)

    xgrams = ([(create_ngrams(clean_text, i)) for i in range(1, 6)])
    stopword_ngrams = copy.deepcopy(xgrams)

    # Filter out ngrams that consist wholly of stopwords
    for i in range(0, 5):
        xgrams[i] = [gram for gram in xgrams[i] if not set(
            twokenize.tokenizeRawTweetText(gram)).issubset(stop_list)]

    # Filter out ngrams that don't consist wholly of stopwords
    for i in range(0, 5):
        stopword_ngrams[i] = [gram for gram in stopword_ngrams[i] if set(
            twokenize.tokenizeRawTweetText(gram)).issubset(stop_list)]

    # Flatten list of lists
    stopword_ngrams = list(chain.from_iterable(stopword_ngrams[1:]))

    # Check the number of HS keyword instances
    grams = list(chain.from_iterable(xgrams))
    hs_keywords_intersect = hs_keywords.intersection(set(grams))

    return [terms_only, list(stopwords_only), hashtags, mentions,
            list(hs_keywords_intersect), xgrams, stopword_ngrams]


def prep_tweet_body(tweet_obj, args, processed_text):
    """ Format the incoming tweet

    Args:
        tweet_obj (dict): Tweet to preprocess.
        args (list): Various datafields to append to the object.
            0: subj_sent_check (bool): Check for subjectivity and sentiment.
            1: subjectivity (num): Subjectivity result.
            2: sentiment (dict): Sentiment result.
        processed_text (list): List of tokens and ngrams etc.

    Returns:
        dict: Tweet with formatted fields
    """

    subj_sent_check = args[0]
    result = tweet_obj

    if subj_sent_check:
        subjectivity = args[1]
        sentiment = args[2]
        result["subjectivity"] = subjectivity
        result["compound_score"] = sentiment["compound"]
        result["neg_score"] = sentiment["neg"]
        result["neu_score"] = sentiment["neu"]
        result["pos_score"] = sentiment["pos"]

    result["hs_keyword_count"] = len(processed_text[4])
    result["hs_keyword_matches"] = processed_text[4]
    result["tokens"] = processed_text[0]
    result["stopwords"] = processed_text[1]
    result["hashtags"] = processed_text[2]
    result["user_mentions"] = processed_text[3]
    result["unigrams"] = processed_text[5][0]
    result["bigrams"] = processed_text[5][1]
    result["trigrams"] = processed_text[5][2]
    result["quadgrams"] = processed_text[5][3]
    result["pentagrams"] = processed_text[5][4]
    result["stopword_ngrams"] = processed_text[6]

    return result


def create_ngrams(text_list, length):
    """ Create ngrams of the specified length from a string of text
    Args:
        text_list   (list): Pre-tokenized text to process.
        length      (int):  Length of ngrams to create.
    """

    clean_tokens = [token for token in text_list if token not in PUNCTUATION]
    return [" ".join(i for i in ngram) for ngram in ngrams(clean_tokens, length)]


def do_create_ngram_collections(text, args):
    """ Create and return ngram collections and set intersections.
    Text must be lowercased if required.
    Args:
        text   (str): Text to process.
        args      (list): Can contains the following:
            0: porn_black_list (list): List of porn keywords.
            1: hs_keywords (list) HS corpus.
            2: black_list  (list) Custom words to filter on.
    """

    porn_black_list = args[0]
    hs_keywords = args[1]
    if args[2]:
        black_list = args[2]

    tokens = twokenize.tokenizeRawTweetText(text)
    unigrams = create_ngrams(tokens, 1)
    # bigrams = create_ngrams(tokens, 2)
    trigrams = create_ngrams(tokens, 3)
    quadgrams = create_ngrams(tokens, 4)

    xgrams = trigrams + quadgrams
    unigrams = set(unigrams)
    xgrams = set(xgrams)

    # Set operations are faster than list iterations.
    # Here we perform a best effort series of filters
    # to ensure we only get tweets we want.
    # unigram_intersect = set(porn_black_list).intersection(unigrams)
    xgrams_intersect = porn_black_list.intersection(xgrams)
    hs_keywords_intersect = hs_keywords.intersection(unigrams)

    if args[2]:
        black_list_intersect = black_list.intersection(unigrams)
    else:
        black_list_intersect = None

    return [None, xgrams_intersect, hs_keywords_intersect, black_list_intersect]


def parallel_preprocess(tweet_list, hs_keywords, subj_check, sent_check):
    """Passes the incoming raw tweets to our preprocessing function.

    Args:
        tweet_list  (list): List of raw tweet texts to preprocess.
        tweet_split (list): List of tokens in the tweet text.
        hs_keywords (set): Keywords to match on.
        subj_check (bool): Check text for subjectivity.
        sent_check (bool): Check text for sentiment.

    Returns:
        list: List of vectorized tweets.
    """
    # num_cores = cpu_count()
    results = Parallel(n_jobs=1)(
        delayed(preprocess_tweet)(tweet, hs_keywords, [subj_check, sent_check]) for tweet in tweet_list)
    return results


def is_garbage(raw_text, precision):
    """ Check if a tweet consists primarly of hashtags, mentions or urls

    Args:
        tweet_obj  (dict): Tweet to preprocess.
    """

    word_list = raw_text.split()
    garbage_check = [
        token for token in word_list if not token.startswith(("#", "@", "http"))]
    garbage_check = " ".join(garbage_check)

    if ((len(raw_text) - len(garbage_check)) / len(raw_text)) >= precision:
        return True
    else:
        return False


def clean_tweet_text(raw_text):
    """Clean up tweet text and preserve emojis"""

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
    for token in clean_text:
        for _match in twokenize.Protected.finditer(token):
            emoji.append(token)

    # If the token is not in the emoji dict, lowercase it
    # TODO preserve tokens with all uppercase chars as a measure of intensity
    emoji = dict.fromkeys(emoji)
    for token in terms_single.copy():
        if not token in emoji:
            terms_single[token.lower()] = terms_single.pop(token)

    return [clean_text, terms_single, emoji]


def remove_urls(raw_text):
    """ Removes urls from text

    Args:
        raw_text (str): Text to filter.
    """
    return re.sub(
        r"(?:http|https):\/\/((?:[\w-]+)(?:\.[\w-]+)+)(?:[\w.,@?^=%&amp;:\/~+#-]*[\w@?^=%&amp;\/~+#-])?", "", raw_text)


def get_emotion_coverage(raw_text):
    """Send the text through an emotion API and return the results"""
    emotion_detector = EmotionDetection.EmotionDetection()
    result = emotion_detector.get_emotion_json(raw_text)
    if result["ambiguous"] == "no":
        return result["groups"]
