# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test file_ops module
"""

import string
from nose.tools import *
from context import hatespeech_core


class TestFileOps(object):
    """ init class """

    def __init__(self):
        self.test_list = ["hello", "world", "how", "are", "you"]

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    def test_build_query_string(self):
        """This method tests the OR concatenation function"""
        result_string = "hello OR world OR how OR are OR you"
        response_string = hatespeech_core.file_ops.build_query_string(
            self.test_list)
        print(response_string)
        assert_equals(response_string, result_string)

    def test_file_operations(self):
        """Combined test for file based ops
        """

        file_list = hatespeech_core.file_ops.get_json_filenames(
            hatespeech_core.settings.JSON_PATH)
        hatespeech_core.file_ops.extract_corpus(file_list)
        response = hatespeech_core.file_ops.read_csv_file("about_sexual_orientation_eng_pg1",
                                                          hatespeech_core.settings.CSV_PATH)
        result = hatespeech_core.file_ops.build_query_string(response)
        assert_equals(len(result), 67)
