# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
Core module
"""

from modules.preprocessing import db_cleaning
from modules.preprocessing import candidate_selection
from modules.preprocessing import feature_prep
from modules.preprocessing import neural_embeddings


def main():
    """
    Run operations
    """
    # candidate_selection.sentiment_pipeline()
    # db_cleaning.preprocessing_pipeline()
    feature_prep.start_feature_extraction()
    # feature_prep.start_store_preprocessed_text()
    # neural_embeddings.train_embeddings()
    # neural_embeddings.train_fasttext_classifier()


if __name__ == "__main__":
    main()
