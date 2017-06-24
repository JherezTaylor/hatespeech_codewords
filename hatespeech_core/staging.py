# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Staging module, just quick functions for checking stuff"""

import string
from pprint import pprint
from itertools import chain
from modules.utils import model_helpers
from modules.utils import word_enrichment
from modules.utils import file_ops
from modules.utils import settings
from modules.utils import text_preprocessing
from modules.db import mongo_base
from modules.db import mongo_search_pipelines
from modules.db import elasticsearch_base
from modules.preprocessing import neural_embeddings
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


def profile_codeword_selection():
    _es = elasticsearch_base.connect(settings.ES_URL)
    positive_hs_filter = "_exists_:hs_keyword_matches"
    negative_hs_filter = "!_exists_:hs_keyword_matches"
    hs_keywords = set(file_ops.read_csv_file(
        "refined_hs_keywords", settings.TWITTER_SEARCH_PATH))

    test_size = 100000
    min_doc_count = 5

    subset_sizes = elasticsearch_base.get_els_subset_size(
        _es, "manchester_event", "hs_keyword_matches")
    doc_count = subset_sizes["positive_count"] + subset_sizes["negative_count"]

    subset_sizes = elasticsearch_base.get_els_subset_size(
        _es, "dailystormer", "hs_keyword_matches")
    doc_count = subset_sizes["positive_count"] + subset_sizes["negative_count"]

    dailystormer_pos_subset = elasticsearch_base.aggregate(
        _es, "dailystormer", "tokens.keyword", False, positive_hs_filter, size=test_size, min_doc_count=min_doc_count)
    dailystormer_neg_subset = elasticsearch_base.aggregate(
        _es, "dailystormer", "tokens.keyword", False, negative_hs_filter, size=test_size, min_doc_count=min_doc_count)

    dailystormer_pos_hs_freqs, dailystormer_pos_vocab_freqs, dailystormer_pos_hs_idfs, dailystormer_pos_vocab_idfs = model_helpers.get_els_word_weights(
        dailystormer_pos_subset[0], doc_count, hs_keywords)
    _, dailystormer_neg_vocab_freqs, _, dailystormer_neg_vocab_idfs = model_helpers.get_els_word_weights(
        dailystormer_neg_subset[0], doc_count, hs_keywords)

    test_size = 150000
    min_doc_count = 10

    subset_sizes = elasticsearch_base.get_els_subset_size(
        _es, "unfiltered_stream", "hs_keyword_matches")
    doc_count = subset_sizes["positive_count"] + subset_sizes["negative_count"]

    unfiltered_stream_pos_subset = elasticsearch_base.aggregate(
        _es, "unfiltered_stream", "tokens.keyword", False, positive_hs_filter, size=test_size, min_doc_count=min_doc_count)
    unfiltered_stream_neg_subset = elasticsearch_base.aggregate(
        _es, "unfiltered_stream", "tokens.keyword", False, negative_hs_filter, size=test_size, min_doc_count=min_doc_count)

    unfiltered_stream_pos_hs_freqs, unfiltered_stream_pos_vocab_freqs, unfiltered_stream_pos_hs_idfs, unfiltered_stream_pos_vocab_idfs = model_helpers.get_els_word_weights(
        unfiltered_stream_pos_subset[0], doc_count, hs_keywords)
    _, unfiltered_stream_neg_vocab_freqs, _, unfiltered_stream_neg_vocab_idfs = model_helpers.get_els_word_weights(
        unfiltered_stream_neg_subset[0], doc_count, hs_keywords)

    dep_model_ids = [0, 7]
    dep_embeddings = neural_embeddings.get_embeddings(
        "dep2vec", model_ids=dep_model_ids, load=True)
    if dep_embeddings:
        dep2vec_dstormer = dep_embeddings[0] if dep_embeddings[0] else None
        dep2vec_ustream = dep_embeddings[1] if dep_embeddings[1] else None

    word_model_ids = [3, 9]
    word_embeddings = neural_embeddings.get_embeddings(
        "ft", model_ids=word_model_ids, load=True)
    if word_embeddings:
        ft_dstormer = word_embeddings[0] if word_embeddings[0] else None
        ft_ustream = word_embeddings[1] if word_embeddings[1] else None

    candidate_codewords = word_enrichment.select_candidate_codewords(biased_embeddings=[dep2vec_dstormer, ft_dstormer],
                                                                     unbiased_embeddings=[dep2vec_ustream, ft_ustream], freq_vocab_pair=[
                                                                         dailystormer_neg_vocab_freqs, unfiltered_stream_neg_vocab_freqs],
                                                                     idf_vocab_pair=[dailystormer_neg_vocab_idfs, unfiltered_stream_neg_vocab_idfs], topn=5, p_at_k_threshold=0.2, hs_keywords=hs_keywords, hs_check=True)

    # pprint(candidate_codewords)


def main():
    """Run operations"""
    profile_codeword_selection()

if __name__ == "__main__":
    main()
