"""Test constants module
"""

from .. context import hatespeech_core

def test_environ_vars():
    """test"""
    assert isinstance(hatespeech_core.constants.JSON_PATH, str)
    assert isinstance(hatespeech_core.constants.CSV_PATH, str)
    assert isinstance(hatespeech_core.constants.DATA_PATH, str)
    assert isinstance(hatespeech_core.constants.DB_URL, str)
    assert isinstance(hatespeech_core.constants.PUSHBULLET_API_KEY, str)
    assert isinstance(hatespeech_core.constants.HASHTAGS, str)
    assert isinstance(hatespeech_core.constants.USER_MENTIONS, str)
    assert isinstance(hatespeech_core.constants.USER_MENTIONS_LIMIT, int)
    assert isinstance(hatespeech_core.constants.HASHTAG_LIMIT, int)
