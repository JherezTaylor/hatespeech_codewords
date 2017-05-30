# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with preparing text for the various neural
embedding approaches as well as training ther models.
"""
from ..db import mongo_base
from ..utils import text_preprocessing
from ..utils import model_helpers
from ..utils import settings


def create_dep_embedding_input(connection_params, filename):
    """ Read and write the data from a conll collection"""
    client = mongo_base.connect()
    connection_params.insert(0, client)

    query = {}
    query["filter"] = {}
    query["projection"] = {"conllFormat": 1, "_id": 0}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    cursor = mongo_base.finder(connection_params, query, False)
    text_preprocessing.prep_conll_file(cursor, filename)


def create_word_embedding_input(connection_params, filename):
    """ Call a collection and write it to disk
    """
    client = mongo_base.connect()
    connection_params.insert(0, client)

    query = {}
    query["filter"] = {"text": {"$ne": None}}
    query["projection"] = {"text": 1, "_id": 0}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    cursor = mongo_base.finder(connection_params, query, False)
    text_preprocessing.prep_word_embedding_file(cursor, filename)


def train_embeddings():
    connection_params_0 = ["twitter", "candidates_hs_exp6_combo_3_Mar_9813004"]
    # connection_params_1 = ["twitter", "hs_candidates_exp6_conll"]
    # write_conll_file(connection_params_1, "hs_candidates_exp6_conll")
    # create_word_embedding_input(
    #     connection_params_0, "word_embedding_hs_exp6.txt")

    model_helpers.train_fasttext_model(
        settings.EMBEDDING_INPUT + "word_embedding_hs_exp6.txt", settings.EMBEDDING_MODELS + "fasttext_hs_exp6")

    model_helpers.train_word2vec_model(
        settings.EMBEDDING_INPUT + "word_embedding_hs_exp6.txt", settings.EMBEDDING_MODELS + "word2vec_hs_exp6")
