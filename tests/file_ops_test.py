# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test file_ops module
"""

import string
from nose.tools import *
from context import hatespeech_core
from nltk.corpus import stopwords


class TestFileOps(object):
    """ init class """

    def __init__(self):
        self.pushbullet_message_title = "Test"
        self.pushbullet_message_body = "Test"
        self.test_list = ["hello", "world", "how", "are", "you"]
        self.sub_sent_string = "Today only kinda sux! But I'll get by, lol"
        self.ngram_string = "the cat is mad, I'm glad"
        self.tweet_text = " I'm #testing is my code good @dev :) fuck"

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
        print(response_string)
        assert_equals(response_string, result_string)

    def test_file_operations(self):
        """Combined test for file based ops
        """

        file_list = hatespeech_core.file_ops.get_filenames(
            hatespeech_core.settings.JSON_PATH)
        hatespeech_core.file_ops.extract_corpus(file_list)
        response = hatespeech_core.file_ops.read_csv_file("about_sexual_orientation_eng_pg1",
                                                          hatespeech_core.settings.CSV_PATH)
        result = hatespeech_core.file_ops.build_query_string(response)
        assert_equals(len(result), 67)

    def test_get_subj_sent(self):
        """Test the subj_sent function.
        """
        result = hatespeech_core.file_ops.get_sent_subj(self.sub_sent_string)
        assert_equals(result[0]["neg"], 0.179)
        assert_equals(result[0]["compound"], 0.2228)
        assert_equals(result[1], 0.85)

    def test_create_ngrams(self):
        """Test create_ngrams function.
        """
        result = hatespeech_core.file_ops.create_ngrams(
            hatespeech_core.file_ops.twokenize.tokenizeRawTweetText(self.ngram_string.lower()), 2)
        assert_equals(result, ["the cat", "cat is",
                               "is mad", "mad i'm", "i'm glad"])
        result = hatespeech_core.file_ops.create_ngrams(
            hatespeech_core.file_ops.twokenize.tokenizeRawTweetText(self.ngram_string.lower()), 1)
        assert_equals(result, ["the", "cat", "is", "mad", "i'm", "glad"])

    def test_prepare_text(self):
        """Test prepare text function.
        """
        punctuation = list(string.punctuation)
        stop_list = dict.fromkeys(stopwords.words(
            "english") + punctuation + ["rt", "via", "RT"])

        result = hatespeech_core.file_ops.prepare_text(
            self.tweet_text.lower(), [stop_list, set(["fuck", "shit"])])
        # Terms only
        assert_equals(sorted(result[0]), sorted(
            ["good", "code", "i'm", ":)", "fuck"]))
        # Stopwords
        assert_equals(sorted(result[1]), sorted(["is", "my"]))
        # Hashtags
        assert_equals(result[2], ["#testing"])
        # Mentions
        assert_equals(result[3], ["@dev"])
        # HS keyword count
        assert_equals(len(result[4]), 1)
        # Ngrams
        assert_equals(sorted(result[5][1]), sorted(["i'm #testing", "#testing is",
                                                    "my code", "code good", "good @dev", "@dev :)", ":) fuck"]))

    def test_emotion_coverage(self):
        """Test emotion API call"""
        result = hatespeech_core.file_ops.get_emotion_coverage(
            self.sub_sent_string)
        assert_not_equal(result, None)
