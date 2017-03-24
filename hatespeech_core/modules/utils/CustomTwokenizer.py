# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Custom tokenizer class
"""

from spacy.tokens import Doc
from . import twokenize
from . import text_preprocessing


class CustomTwokenizer(object):
    """ Custom class for replacing the Penn Treebank tokenizer that spacy uses
    with one built for twitter.
    """

    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        clean_text = text_preprocessing.remove_urls(text)
        words = twokenize.customTokenizeRawTweetText(clean_text)
        return Doc(self.vocab, words=words)
        