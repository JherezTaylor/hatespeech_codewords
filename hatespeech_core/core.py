# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Core module
"""

from modules.preprocessing import db_cleaning
from modules.preprocessing import candidate_selection
from modules.utils import twitter_api

def main():
    """
    Run operations
    """
    # db_cleaning.preprocessing_pipeline()
    # candidate_selection.sentiment_pipeline()
    # print twitter_api.run_status_lookup(["803467419377352706", "803467418530185216"])
    db_cleaning.run_update_missing_text()

if __name__ == "__main__":
    main()
