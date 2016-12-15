from nose.tools import *
import hatespeech_core


def setup():
    print "SETUP!"


def teardown():
    print "TEAR DOWN!"

@with_setup(setup, teardown)
def test_basic():
    print "I RAN!"

def test_sum():
    eq_(2+2,5)
