# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""Test text_preprocessing module
"""

import string
from nose.tools import *
from context import hatespeech_core
from nltk.corpus import stopwords


class TestFileOps(object):
    """ init class """

    def __init__(self):
        self.test_list = ["hello", "world", "how", "are", "you"]
        self.sub_sent_string = "Today only kinda sux! But I'll get by, lol"
        self.ngram_string = "the cat is mad, I'm glad"
        self.tweet_text = " I'm #testing is my code good @dev :) fuck"
        self.url_load = "RT @Alt_Right_: #AltNews Feminist Sluts Pissed They’re Getting Judo-Memed with Grab A... https://t.co/QDCToMErC6 #AltRight #MAGA https://t…"

    def setup(self):
        """This method is run once before _each_ test method is executed"""

    def teardown(self):
        """This method is run once after _each_ test method is executed"""

    # def test_remove_urls(self):
    #     """Test the remove url function"""
    #     result = hatespeech_core.text_preprocessing.remove_urls(self.url_load)
    #     print(result)

    def test_get_subj_sent(self):
        """Test the subj_sent function.
        """
        result = hatespeech_core.text_preprocessing.get_sent_subj(
            self.sub_sent_string)
        assert_equals(result[0]["neg"], 0.179)
        assert_equals(result[0]["compound"], 0.2228)
        assert_equals(result[1], 0.85)

    def test_create_ngrams(self):
        """Test create_ngrams function.
        """
        result = hatespeech_core.text_preprocessing.create_ngrams(
            hatespeech_core.text_preprocessing.twokenize.tokenizeRawTweetText(self.ngram_string.lower()), 2)
        assert_equals(result, ["the cat", "cat is",
                               "is mad", "mad i'm", "i'm glad"])
        result = hatespeech_core.text_preprocessing.create_ngrams(
            hatespeech_core.text_preprocessing.twokenize.tokenizeRawTweetText(self.ngram_string.lower()), 1)
        assert_equals(result, ["the", "cat", "is", "mad", "i'm", "glad"])

    def test_prepare_text(self):
        """Test prepare text function.
        """
        punctuation = list(string.punctuation)
        stop_list = dict.fromkeys(stopwords.words(
            "english") + punctuation + ["rt", "via", "RT"])

        result = hatespeech_core.text_preprocessing.prepare_text(
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
        # Clean text
        assert_equals(result[7], ["i'm", '#testing',
                                  'code', 'good', '@dev', ':)', 'fuck'])

    @nottest
    def test_emotion_coverage(self):
        """Test emotion API call"""
        result = hatespeech_core.text_preprocessing.get_emotion_coverage(
            self.sub_sent_string, "text")
        assert_not_equal(result, None)
