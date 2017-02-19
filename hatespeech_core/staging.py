# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Staging module, just quick functions
"""

import string
from modules.utils import file_ops
from modules.utils import settings
from nltk.corpus import words
from nltk.corpus import stopwords


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
    bigrams = file_ops.create_ngrams(text, 2)
    bigrams = [ngram for ngram in bigrams if not set(
        file_ops.twokenize.tokenizeRawTweetText(ngram)).issubset(set(stop_list))]
    print(bigrams)


def main():
    """
    Run operations
    """
    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    check_token_lengths(porn_black_list)
    ngram_stopword_check("is to hello world boss hi is")
if __name__ == "__main__":
    main()
