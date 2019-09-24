# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Custom tokenizer class
"""

import string
from nltk.corpus import stopwords
from spacy.tokens import Doc
from . import twokenize
from . import text_preprocessing

PUNCTUATION = list(string.punctuation)
STOP_LIST = set(stopwords.words(
    "english") + PUNCTUATION + ["rt", "via", "RT", "..."])


class CustomTwokenizer(object):
    """ Custom class for replacing the Penn Treebank tokenizer that spacy uses
    with one built for twitter.
    Removes urls, normalizes usermentions and tokenizes using ARK Twokenizer
    """

    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        clean_text = text_preprocessing.remove_urls(text)
        words = twokenize.customTokenizeRawTweetText(clean_text)
        return Doc(self.vocab, words=words)


class EmbeddingTwokenizer(object):
    """ Custom class for replacing the Penn Treebank tokenizer that spacy uses
    with one built for twitter.
    Removes URLS, numbers, and stopwords, normalizes @usermentions and tokenizes
    using ARK Twokenizer.
    """

    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        clean_text = text_preprocessing.remove_urls(text)
        words = twokenize.customTokenizeRawTweetText(clean_text)
        result = [token for token in words if len(
            token.strip(string.digits)) == len(token) and token not in STOP_LIST]
        return Doc(self.vocab, words=result)
