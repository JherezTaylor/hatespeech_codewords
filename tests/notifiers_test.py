# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test file_ops module
"""

from nose.tools import *
from context import hatespeech_core

class TestNotifiers(object):
    """ init class """

    def __init__(self):
        self.pushbullet_message_title = "Test"
        self.pushbullet_message_body = "Test"

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    # @nottest
    def test_send_job_notification(self):
        """This method tests the pushbullet notification function"""
        response = hatespeech_core.notifiers.send_job_notification(
            self.pushbullet_message_title, self.pushbullet_message_body)
        assert_equals(response.status_code, 200)
