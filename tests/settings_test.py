# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""Test settings module
"""

from context import hatespeech_core

def test_settings():
    """Test that settings are loaded"""
    assert isinstance(hatespeech_core.settings.JSON_PATH, str)
    assert isinstance(hatespeech_core.settings.CSV_PATH, str)
    assert isinstance(hatespeech_core.settings.DATA_PATH, str)
    assert isinstance(hatespeech_core.settings.DB_URL, str)
    assert isinstance(hatespeech_core.settings.PUSHBULLET_API_KEY, str)
    assert isinstance(hatespeech_core.settings.HASHTAGS, str)
    assert isinstance(hatespeech_core.settings.USER_MENTIONS, str)
    assert isinstance(hatespeech_core.settings.USER_MENTIONS_LIMIT, int)
    assert isinstance(hatespeech_core.settings.HASHTAG_LIMIT, int)