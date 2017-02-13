# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Staging module, just quick functions
"""

from modules.utils import file_ops
from modules.utils import settings
from nltk.corpus import words
from pprint import pprint


def check_token_lengths(wordlist):
    """Find the number of unigrams in the blacklist"""

    unigrams = [word for word in wordlist if len(
        file_ops.twokenize.tokenizeRawTweetText(word)) == 1]
    result = unusual_words(unigrams)
    print "Single token count: {0}".format(len(unigrams))
    print "Non dictionary matches: {0}".format(len(result[0]))
    print "Dictionary matches: {0}".format(len(result[1]))


def unusual_words(text_list):
    """ Filtering a Text: this program computes the vocabulary of a text,
    then removes all items that occur in an existing wordlist,
    leaving just the uncommon or mis-spelt words."""
    # text_vocab = set(w.lower() for w in text_list if w.isalpha())
    text_vocab = set(w.lower() for w in text_list)
    english_vocab = set(w.lower() for w in words.words())
    unusual = text_vocab - english_vocab
    return [sorted(unusual), sorted(text_vocab - unusual)]


def main():
    """
    Run operations
    """
    porn_black_list = dict.fromkeys(file_ops.read_csv_file(
        "porn_blacklist", settings.WORDLIST_PATH))

    check_token_lengths(porn_black_list)

if __name__ == "__main__":
    main()
