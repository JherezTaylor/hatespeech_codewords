# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

from nose.tools import *
from context import hatespeech_core


class TestMongoMethods(object):
    """ init class """

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    # @nottest
    def test_connection(self):
        """This method tests the mongo db connection"""
        conn = hatespeech_core.mongo_methods.connect()
        ping = conn.admin.command('ping')
        assert ping >= 1
        