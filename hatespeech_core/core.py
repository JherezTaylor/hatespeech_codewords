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
    # candidate_selection.sentiment_pipeline()
    # classifier_concept.start_feature_extraction()
    # db_cleaning.preprocessing_pipeline()
    # classifier_concept.train_embeddings()

if __name__ == "__main__":
    main()
