# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 2.7

"""
Core module
"""

from modules.preprocessing import db_cleaning
from modules.preprocessing import candidate_selection

def main():
    """
    Run operations
    """
    # db_cleaning.preprocessing_pipeline()
    candidate_selection.sentiment_pipeline()


if __name__ == "__main__":
    main()
