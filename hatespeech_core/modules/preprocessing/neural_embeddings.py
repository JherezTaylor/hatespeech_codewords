# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module calls functions that deal with preparing text for the various neural
embedding approaches as well as training ther models.
"""

import glob
import os
import fileinput
from joblib import Parallel, delayed, cpu_count
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
    query["filter"] = {}
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
        partitions[idx] = (partitions[idx][0], partitions[idx][1], idx)

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

    job_list = [
        ["twitter_annotated_datasets",
            "NAACL_SRW_2016_features", "word_embedding_NACCL"],
        ["twitter_annotated_datasets",
         "NLP_CSS_2016_expert_features", "word_embedding_NLP_CSS"],
        ["twitter_annotated_datasets", "crowdflower_features", "word_embedding_crwdflr"],
        ["dailystormer_archive", "d_stormer_documents",
            "word_embedding_daily_stormer"],
        ["twitter", "melvyn_hs_users", "word_embedding_melvyn_hs"],
        ["manchester_event", "tweets", "word_embedding_daily_manchester"],
        ["inauguration", "tweets", "word_embedding_inauguration"],
        ["uselections", "tweets", "word_embedding_uselections"],
        ["twitter", "candidates_hs_exp6_combo_3_Mar_9813004", "word_embedding_hs_exp6"],
        ["unfiltered_stream_May17", "tweets", "word_embedding_unfiltered_stream"],
        ["twitter", "tweets", "word_embedding_twitter"]
    ]
    # Prep data
    for job in job_list:
        create_word_embedding_input(job, job[2])

    # # Train model
    # for job in job_list:
    #     model_helpers.train_fasttext_model(
    #         settings.EMBEDDING_INPUT + job[2] + ".txt", settings.EMBEDDING_MODELS + "fasttext_" + job[2])
    #     model_helpers.train_word2vec_model(
    # settings.EMBEDDING_INPUT + job[2] + ".txt", settings.EMBEDDING_MODELS +
    # "word2vec_" + job[2])
