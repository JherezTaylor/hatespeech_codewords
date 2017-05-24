# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Core module
"""

from modules.preprocessing import db_cleaning
from modules.preprocessing import candidate_selection
from modules.preprocessing import classifier_concept
from modules.utils import twitter_api


def main():
    """
    Run operations
    """
    # db_cleaning.preprocessing_pipeline()
    # candidate_selection.sentiment_pipeline()
    # db_cleaning.run_update_missing_text()
    # classifier_concept.start_feature_extraction()

if __name__ == "__main__":
    main()
