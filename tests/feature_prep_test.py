# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test feature_prep module
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
        nlp = hatespeech_core.feature_prep.init_nlp_pipeline()
        result_set = [("I'm", 'NN'), ('here', 'RB'), ('get', 'VB'),
                      ('rekt', 'NN'), ('#squadgoals', 'NNS'), ('okay', 'JJ')]
        response_string = hatespeech_core.feature_prep.extract_lexical_features_test(nlp,
                                                                                     self.test_list)
        assert_equals(response_string, result_set)
