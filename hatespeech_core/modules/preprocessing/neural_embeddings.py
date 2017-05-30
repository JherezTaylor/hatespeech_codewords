# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with preparing text for the various neural
embedding approaches as well as training ther models.
"""

import glob
import os
from joblib import Parallel, delayed, cpu_count
import fileinput
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

    query = {}
    query["filter"] = {"preprocessed_txt": {"$ne": None}}
    query["projection"] = {"preprocessed_txt": 1, "_id": 0}
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True

    connection_params.insert(0, client)
    collection_size = mongo_base.finder(connection_params, query, True)
    del connection_params[0]
    client.close()

    num_cores = cpu_count()
    partition_size = collection_size // num_cores
    partitions = [(i, partition_size)
                  for i in range(0, collection_size, partition_size)]
    # Account for lists that aren't evenly divisible, update the last tuple to
    # retrieve the remainder of the items

    partitions[-1] = (partitions[-1][0], (collection_size - partitions[-1][0]))

    for idx, _partition in enumerate(partitions):
        partitions[idx] = (partitions[idx][0], partitions[idx][0], idx)

    Parallel(n_jobs=num_cores)(delayed(text_preprocessing.prep_word_embedding_file)(
        connection_params, query, partition, filename) for partition in partitions)

    # Merge all files
    filenames = glob.glob(settings.EMBEDDING_INPUT + filename + "*.txt")
    with open(settings.EMBEDDING_INPUT + filename + ".txt", 'w') as fout, fileinput.input(filenames) as fin:
        for line in fin:
            fout.write(line)
    # Clean up files
    for _f in filenames:
        os.remove(_f)


def train_embeddings():
    """ Start job
    """
    # connection_params_1 = ["twitter", "hs_candidates_exp6_conll"]
    # write_conll_file(connection_params_1, "hs_candidates_exp6_conll")
    connection_params_0 = ["twitter", "candidates_hs_exp6_combo_3_Mar_9813004"]
    create_word_embedding_input(
        connection_params_0, "word_embedding_hs_exp6")

    # model_helpers.train_fasttext_model(
    # settings.EMBEDDING_INPUT + "word_embedding_hs_exp6.txt",
    # settings.EMBEDDING_MODELS + "fasttext_hs_exp6")

    # model_helpers.train_word2vec_model(
    # settings.EMBEDDING_INPUT + "word_embedding_hs_exp6.txt",
    # settings.EMBEDDING_MODELS + "word2vec_hs_exp6")
