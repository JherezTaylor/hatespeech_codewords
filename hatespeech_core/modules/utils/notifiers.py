# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various io functions for use throughout the project
"""
from time import time
import cProfile
import pstats
import requests
import ujson
from . import settings


def timing(func):
    """Decorator for timing run time of a function
    """

    def wrap(*args):
        """Wrapper
        """
        time1 = time()
        ret = func(*args)
        time2 = time()
        settings.logger.debug("%s function took %0.3f ms",
                              func.__name__, (time2 - time1) * 1000.0)
        return ret

    return wrap


def do_cprofile(func):
    """Decorator for profiling a function
    """

    def profiled_func(*args, **kwargs):
        """Wrapper
        """
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            stats = pstats.Stats(profile)
            stats.sort_stats("time").print_stats(20)

    return profiled_func


@timing
def send_job_notification(title, body):
    """ Send a notification via Pushbullet.

     Args:
        json_obj (json_obj).

    Indicates whether a job has completed or whether an error occured.
    """
    headers = {"Access-Token": settings.PUSHBULLET_API_KEY,
               "Content-Type": "application/json"}
    payload = {"type": "note", "title": title, "body": ujson.dumps(body)}
    url = "https://api.pushbullet.com/v2/pushes"
    return requests.post(url, headers=headers, data=ujson.dumps(payload))


def send_job_completion(run_time, args):
    """Format and print the details of a completed job

    Args:
        run_time (list): Start and end times.
        args (list): Contains the following:
            0: function_name (str): Name of the function that was run.
            1: message_text  (str): Text to be sent in notification.
    """

    time_diff = round((run_time[1] - run_time[0]), 2)
    settings.logger.debug("%s function took %0.3f seconds", args[0], time_diff)
    send_notification = send_job_notification(
        settings.MONGO_SOURCE + ": " + args[1] + " took " + str(time_diff) + " seconds", "Complete")
    settings.logger.debug(send_notification.content)
