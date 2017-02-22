# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Staging module, just quick functions
"""

import string
from modules.utils import file_ops
from modules.utils import settings
from modules.db import mongo_base
from modules.db import mongo_search_pipelines
from nltk.corpus import words
from nltk.corpus import stopwords
import threading
from pprint import pprint


def check_token_lengths(wordlist):
    """Find the number of unigrams in the blacklist"""

    unigrams = [word for word in wordlist if len(
        file_ops.twokenize.tokenizeRawTweetText(word)) == 2]
    result = unusual_words(unigrams)
    print("Single token count: {0}".format(len(unigrams)))
    print("Non dictionary matches: {0}".format(len(result[0])))
    print("Dictionary matches: {0}".format(len(result[1])))
    # pprint(result[0])


def unusual_words(text_list):
    """Filtering a Text: this program computes the vocabulary of a text,
    then removes all items that occur in an existing wordlist,
    leaving just the uncommon or mis-spelt words."""
    # text_vocab = set(w.lower() for w in text_list if w.isalpha())
    text_vocab = set(w.lower() for w in text_list)
    english_vocab = set(w.lower() for w in words.words())
    unusual = text_vocab - english_vocab
    return [sorted(unusual), sorted(text_vocab - unusual)]


def ngram_stopword_check(text):
    """Check if all the tokens in an ngram are stopwords"""
    punctuation = list(string.punctuation)
    stop_list = dict.fromkeys(stopwords.words(
        "english") + punctuation + ["rt", "via", "RT"])
    bigrams = file_ops.create_ngrams(file_ops.twokenize.tokenizeRawTweetText(text.lower()), 2)
    bigrams = [ngram for ngram in bigrams if not set(
        file_ops.twokenize.tokenizeRawTweetText(ngram)).issubset(set(stop_list))]
    print(bigrams)


def test_parallel_scan(client):
    """ Test pymongo parallel scan
    """

    connection_params = [client, "twitter_test", "test_suite"]

    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]

    cursors = dbo[collection].parallel_scan(20)
    threads = [threading.Thread(
        target=process_cursor, args=(cursor,)) for cursor in cursors]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def process_cursor(cursor):
    """ Thread safe process
    """
    documents = [str(document["_id"]) for document in cursor]
    print(len(documents))
    # for document in cursor:
    #     pprint(str(document["_id"]))


@file_ops.do_cprofile
def runner(length):
    """ Start job
    """
    # 170 seconds
    client = mongo_base.connect()
    for _ in range(length):
        test_parallel_scan(client)


@file_ops.do_cprofile
def test_linear_scan(length):
    """ Test linear scan
    """
    # 181 Seconds
    client = mongo_base.connect()
    connection_params = [client, "twitter", "tweets"]

    db_name = connection_params[1]
    collection = connection_params[2]
    dbo = client[db_name]

    for _ in range(length):
        cursor = dbo[collection].find({}, {"id_str":1}).skip(6500000).limit(100)
        # TODO Get collection count, split into 5 chunks and create n cursors
        process_cursor(cursor)

def test_pipeline():
    """ Test pipeline performance
    """
    client = mongo_base.connect()
    connection_params = [client, "twitter_test", "test_performance"]
    args2 = [False, "perf_output", False, False]
    mongo_search_pipelines.select_hs_candidates(connection_params, args2)

def main():
    """
    Run operations
    """
    # porn_black_list = dict.fromkeys(file_ops.read_csv_file(
    #     "porn_blacklist", settings.WORDLIST_PATH))

    # check_token_lengths(porn_black_list)
    # ngram_stopword_check("is to hello world boss hi is")
    # runner(1)
    # test_linear_scan(1)
    test_pipeline()
    # print(file_ops.twokenize.tokenizeRawTweetText("hello world #gtg"))
if __name__ == "__main__":
    main()
