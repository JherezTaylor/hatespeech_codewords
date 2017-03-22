# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test classifier_concept module
"""

import string
from nose.tools import *
from context import hatespeech_core


class TestFileOps(object):
    """ init class """

    def __init__(self):
        self.test_list = ["I'm here", "get rekt", "#squadgoals okay"]

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_extract_lexical_features(self):
        """This method tests the OR concatenation function"""
        result_set = [("I'm", 'NNS'), ('here', 'RB'), ('get', 'VB'),
                      ('rekt', 'JJ'), ('#squadgoals', 'NNS'), ('okay', 'JJ')]
        response_string = hatespeech_core.classifier_concept.extract_lexical_features_test(
            self.test_list)
        assert_equals(response_string, result_set)

