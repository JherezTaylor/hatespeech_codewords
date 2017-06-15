# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test elasticsearch_base module
"""

from nose.tools import *
from context import hatespeech_core


class TestMongoMethods(object):
    """ init class """

    def __init__(self):
        self.client = hatespeech_core.elasticsearch_base.connect()

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_connection(self):
        """Test elasticsearch connection"""
        ping = self.client.ping()
        assert ping
