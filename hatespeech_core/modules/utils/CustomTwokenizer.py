# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Custom tokenizer class
"""

from spacy.tokens import Doc
from . import twokenize


class CustomTwokenizer(object):
    """ Custom class for replacing the Penn Treebank tokenizer that spacy uses
    with one built for twitter.
    """

    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        words = twokenize.tokenizeRawTweetText(text)
        return Doc(self.vocab, words=words)
        