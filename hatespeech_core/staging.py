# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Staging module, just quick functions for checking stuff"""

import string
from pprint import pprint
from itertools import chain
from modules.utils import file_ops
from modules.utils import settings
from modules.utils import text_preprocessing
from modules.db import mongo_base
from modules.db import mongo_search_pipelines
from nltk.corpus import words
from nltk.corpus import stopwords
from joblib import Parallel, delayed


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
    bigrams = text_preprocessing.create_ngrams(
        file_ops.twokenize.tokenizeRawTweetText(text.lower()), 2)
    bigrams = [ngram for ngram in bigrams if not set(
        file_ops.twokenize.tokenizeRawTweetText(ngram)).issubset(set(stop_list))]
    print(bigrams)


# @profile
def test_linear_scan(connection_params, sample_size):
    """Test linear scan"""
    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    cursor = dbo[collection].find({}, {"id_str": 1}).limit(sample_size)
    documents = {str(document["_id"]) for document in cursor}
    print(len(documents))


def process_cursor(cursor):
    """Return all documents in a cursor"""
    documents = [str(document["_id"]) for document in cursor]
    return documents

# @profile


def process_partition(partition, connection_params):
    """Thread safe process
    partition stores a tuple with the skip and limit values
    """
    client = mongo_base.connect()
    db_name = connection_params[0]
    collection = connection_params[1]
    dbo = client[db_name]
    dbo.authenticate(settings.MONGO_USER, settings.MONGO_PW,
                     source=settings.DB_AUTH_SOURCE)

    cursor = dbo[collection].find({}, {"id_str": 1}).skip(
        partition[0]).limit(partition[1])
    documents = {str(document["_id"]) for document in cursor}
    return documents


def parallel_test(num_cores, connection_params, sample_size):
    """Test parallel functionality"""

    partition_size = sample_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, sample_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items
    partitions[-1] = (partitions[-1][0], (sample_size - partitions[-1][0]))

    results = Parallel(n_jobs=num_cores)(
        delayed(process_partition)(partition, connection_params) for partition in partitions)
    results = list(chain.from_iterable(results))
    print(partitions)
    print(len(results))


def main():
    """Run operations"""
    # porn_black_list = dict.fromkeys(file_ops.read_csv_file(
    #     "porn_blacklist", settings.WORDLIST_PATH))

    # check_token_lengths(porn_black_list)
    # ngram_stopword_check("is to hello world boss hi is")
    # connection_params = [mongo_base.connect(), "twitter", "test_suite"]
    # sample_size = 10**6
    # connection_params = ["twitter", "tweets"]
    # # test_linear_scan(connection_params, sample_size)
    # parallel_test(2, connection_params, sample_size)

if __name__ == "__main__":
    main()
