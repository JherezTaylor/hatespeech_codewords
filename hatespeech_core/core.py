# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Core module
"""

from modules.preprocessing import db_cleaning
from modules.preprocessing import candidate_selection
from modules.preprocessing import feature_prep
from modules.utils import twitter_api


def main():
    """
    Run operations
    """
    # candidate_selection.sentiment_pipeline()
    # feature_prep.start_feature_extraction()
    # db_cleaning.preprocessing_pipeline()
    # neural_embeddings.train_embeddings()
    feature_prep.start_store_preprocessed_text()


if __name__ == "__main__":
    main()
