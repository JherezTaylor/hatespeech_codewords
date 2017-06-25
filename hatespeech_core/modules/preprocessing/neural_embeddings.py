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
from joblib import Parallel, delayed, cpu_count, dump
from gensim.models import KeyedVectors, Word2Vec
from ..db import mongo_base
from ..utils import text_preprocessing
from ..utils import model_helpers
from ..utils import train_embeddings
from ..utils import file_ops
from ..utils import settings


def load_embedding(filename, embedding_type):
    """ Load a fasttext or word2vec embedding
    Args:
        filename (str)
        embedding_type (str): kv:keyedVectors w2v:word2vec
    """
    if embedding_type == "kv":
        return KeyedVectors.load_word2vec_format(settings.EMBEDDING_MODELS + filename, binary=False, unicode_errors="ignore")
    elif embedding_type == "w2v":
        model = Word2Vec.load(settings.EMBEDDING_MODELS + filename)
        word_vectors = model.wv
        del model
        return word_vectors


def get_embeddings(embedding_type, model_ids=None, load=False):
    """ Helper function for loading embedding models
    Args:
        embedding_type (str): dep2vec,ft:keyedVectors w2v:word2vec
        model_ids (list): List of ints referencing the models.
    Model positions:
        dep2vec_dstormer: 0
        dep2vec_hs_candidates_exp6: 1
        dep2vec_inaug: 2
        dep2vec_manchester: 3
        dep2vec_melvyn_hs: 4
        dep2vec_twitter: 5
        dep2vec_uselec: 6
        dep2vec_ustream: 7

        fasttext_dstormer: 3
        fasttext_hs_candidates_exp6: 4
        fasttext_inaug: 5
        fasttext_manchester: 6
        fasttext_melvyn_hs: 7
        fasttext_twitter: 8
        fasttext_ustream: 9
        fasttext_uselec: 10
    """

    model_format = "kv" if embedding_type == "dep2vec" or embedding_type == "ft" else "w2v"

    if model_ids and load:
        if embedding_type == "dep2vec":
            embeddings_ref = sorted(file_ops.get_model_names(
                glob.glob(settings.EMBEDDING_MODELS + "dim*")))
        elif embedding_type == "ft":
            embeddings_ref = sorted(file_ops.get_model_names(
                glob.glob(settings.EMBEDDING_MODELS + "*.vec")))
        elif embedding_type == "w2v":
            embeddings_ref = sorted(file_ops.get_model_names(
                glob.glob(settings.EMBEDDING_MODELS + "word2vec_*")))

        for idx, ref in enumerate(embeddings_ref):
            print(idx, ref)

        loaded_models = []
        for idx in model_ids:
            loaded_models.append(load_embedding(
                embeddings_ref[idx], model_format))
        return loaded_models
    else:
        print("Embedding models not loaded")


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


def create_word_embedding_input(connection_params, filter, filename):
    """ Call a collection and write it to disk """

    client = mongo_base.connect()

    query = {}
    query["filter"] = {}
    query["projection"] = {"word_embedding_txt": 1, "_id": 0}
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


def create_fasttext_clf_input(connection_params, filename, fieldname, train_size):
    """ Call a collection and write to disk in fasttext classification format"""

    conn = [connection_params[0], connection_params[1]]
    projection = {"_id": 0, fieldname: 1, "annotation": 1}
    _df = model_helpers.fetch_as_df(conn, projection)
    _df["annotation"] = ['__label__' +
                         str(annotation) for annotation in _df.annotation]

    _df[fieldname] = _df[fieldname].str.lower()
    df_train, df_test = model_helpers.train_test_split(
        _df, train_size=train_size)
    df_train[['annotation', fieldname]].to_csv(
        settings.EMBEDDING_INPUT + filename + "_train.txt", index=False, header=None, sep=" ")
    df_test[['annotation', fieldname]].to_csv(
        settings.EMBEDDING_INPUT + filename + "_test.txt", index=False, header=None, sep=" ")

    dump(df_train, settings.EMBEDDING_INPUT + filename +
         "_train.pkl.compressed", compress=True)
    dump(df_test, settings.EMBEDDING_INPUT + filename +
         "_test.pkl.compressed", compress=True)


def train_word_embeddings():
    """ Start embedding tasks
    """

    job_list = [
        ["twitter_annotated_datasets",
            "NAACL_SRW_2016", "embedding_NACCL"],
        ["twitter_annotated_datasets",
         "NLP_CSS_2016_expert", "embedding_NLP_CSS"],
        ["twitter_annotated_datasets", "crowdflower", "embedding_crwdflr"],
        ["dailystormer_archive", "d_stormer_documents",
            "embedding_daily_stormer"],
        ["twitter", "melvyn_hs_users", "embedding_melvyn_hs"],
        ["manchester_event", "tweets", "embedding_manchester"],
        ["inauguration", "tweets", "embedding_inauguration"],
        ["uselections", "tweets", "embedding_uselections"],
        ["twitter", "candidates_hs_exp6_combo_3_Mar_9813004", "embedding_hs_exp6"],
        ["unfiltered_stream_May17", "tweets", "embedding_unfiltered_stream"],
        ["twitter", "tweets", "embedding_twitter"],
        ["inauguration_no_filter", "tweets", "embedding_inauguration_unfiltered"]
    ]

    clean_collection = [
        ["twitter_annotated_datasets",
            "NAACL_SRW_2016"],
        ["twitter_annotated_datasets",
         "NLP_CSS_2016_expert"]
    ]

    # create_word_embedding_input(clean_collection, "clean_test")

    # Prep data
    # for job in job_list:
    #     create_word_embedding_input(job, job[2])

    # Train fasttext and w2v model
    # for job in job_list:
    #     train_embeddings.fasttext_model(
    #         settings.EMBEDDING_INPUT + job[2] + ".txt", settings.EMBEDDING_MODELS + "fasttext_" + job[2])
        # train_embeddings.word2vec_model(
        #     settings.EMBEDDING_INPUT + job[2] + ".txt", settings.EMBEDDING_MODELS +
        #     "word2vec_" + job[2])


def train_dep2vec_model():
    """ Start dependenc2vec classification"""
    dep_job_list = [
        ["dailystormer_archive", "d_stormer_documents_conll",
         "dstormer_conll"],
        ["twitter", "melvyn_hs_users_conll", "melvynhs_conll"],
        ["manchester_event", "tweets_conll", "manch_conll"],
        ["inauguration", "tweets_conll", "inaug_conll"],
        ["uselections", "tweets_conll", "uselec_conll"],
        ["unfiltered_stream_May17", "tweets_conll", "ustream_conll"],
        ["twitter", "tweets_conll", "twitter_conll"]
    ]

    # for job in dep_job_list:
    #     create_dep_embedding_input(job[0:2], job[2])

    for job in dep_job_list:
        train_embeddings.dep2vec_model(job[2], 50, 100, 200)


def train_fasttext_classifier():
    """ Start fasttext classification
    """
    job_list = [
        ["twitter_annotated_datasets",
            "NAACL_SRW_2016_features", "fasttext_clf_NACCL"],
        ["twitter_annotated_datasets",
         "NLP_CSS_2016_expert_features", "fasttext_clf_NLP_CSS"],
        ["twitter_annotated_datasets", "crowdflower_features", "fasttext_clf_crwdflr"]
    ]

    for job in job_list:
        create_fasttext_clf_input(job, job[2], "text", 0.8)

    for job in job_list:
        train_embeddings.fasttext_classifier(
            settings.EMBEDDING_INPUT + job[2] + "_train.txt", settings.CLASSIFIER_MODELS + job[2], epoch=20, dim=200)
