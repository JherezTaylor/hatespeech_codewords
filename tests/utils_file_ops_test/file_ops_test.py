# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Test file_ops module
"""

from nose.tools import *
from ..context import hatespeech_core

class TestFileOps(object):
    @classmethod
    def setup_class(klass):
        """This method is run once for each class before any tests are run"""

    @classmethod
    def teardown_class(klass):
        """This method is run once for each class _after_ all tests are run"""

    def setUp(self):
        """This method is run once before _each_ test method is executed"""
        self.title = 'hey'
        self.body = 'test'

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    # @with_setup(self)
    def test_send_job_notification(self):
        response = hatespeech_core.file_ops.send_job_notification(self.title, self.body)
        assert response is not None
        print response

