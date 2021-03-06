# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test mongo_base module
"""

from nose.tools import *
from context import hatespeech_core


class TestMongoMethods(object):
    """ init class """

    def __init__(self):
        self.client = hatespeech_core.mongo_base.connect()
        self.db_name = "twitter"
        self.collection = "test_suite"
        self.connection_params = [self.client,
                                  "twitter", "test_suite"]
        self.lang_list_length = 2
        self.lang_list = ["en", "und"]
        self.query = {}
        self.query["filter"] = {"lang": "en"}
        self.query["projection"] = {}
        self.query["limit"] = 10
        self.query["skip"] = 0
        self.query["no_cursor_timeout"] = True

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    # @nottest
    def test_connection(self):
        """Test mongo db connection"""
        ping = self.client.admin.command("ping")
        assert 'ok' in ping
        assert ping.get('ok') >= 1

    # @nottest
    def test_get_language_list(self):
        """Test get language list for the tweet collection"""
        result = hatespeech_core.mongo_base.get_language_list(
            self.connection_params)
        assert_equals(len(result), self.lang_list_length)

    # @nottest
    def test_get_language_distribution(self):
        """Test the language distribution of tweets"""
        db_result = hatespeech_core.mongo_base.get_language_distribution(
            self.connection_params, self.lang_list)
        expected = [{"language": "en", "count": 941},
                    {"language": "und", "count": 59}]

        result = list(db_result)
        assert_equals(list(result), expected)

        # Cleanup
        dbo = self.client[self.db_name]
        dbo[self.collection + "_lang_distribution"].drop()

    def test_finder(self):
        """Test query runner"""
        db_result = hatespeech_core.mongo_base.finder(
            self.connection_params, self.query, True)
        assert_equals(db_result, 941)
