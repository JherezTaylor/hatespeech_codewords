# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Test file_ops module
"""

from nose.tools import *
from context import hatespeech_core


class TestFileOps(object):
    """ init class """

    def __init__(self):
        self.pushbullet_message_title = "hey"
        self.pushbullet_message_body = "test"
        self.test_list = ["hello", "world", "how", "are", "you"]

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    @nottest
    def test_send_job_notification(self):
        """This method tests the pushbullet notification function"""
        response = hatespeech_core.file_ops.send_job_notification(
            self.pushbullet_message_title, self.pushbullet_message_body)
        assert_equals(response.status_code, 200)

    def test_build_query_string(self):
        """This method tests the OR concatenation function"""
        result_string = "hello OR world OR how OR are OR you"
        response_string = hatespeech_core.file_ops.build_query_string(
            self.test_list)
        print response_string
        assert_equals(response_string, result_string)
