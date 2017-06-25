# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various functions that are used to enrich the
contextual representation of words.
"""

import numpy as np
from . import settings
from . import notifiers


def gensim_top_k_similar(model, row, field_name, k):
    """ Returns the top k similar word vectors from a word embedding model.
    Args
    ----
        model (gensim.models): Gensim word embedding model.

        row (pandas.Dataframe): Row extracted from a dataframe.

        field_name (str): Name of dataframe column to extract.

        k (int): Number of results to return.
    """

    similar_words = []
    model_vocab = set(model.index2word)
    for word in row[field_name]:
        if word in model_vocab:
            matches = model.similar_by_word(word, topn=k, restrict_vocab=None)
            for _m in matches:
                similar_words.append(_m[0])
    return similar_words


def spacy_top_k_similar(word, k):
    """ Returns the top k similar word vectors from a spacy embedding model.
    Args
    ----
        word (spacy.token): Gensim word embedding model.

        k (int): Number of results to return.
    """
    queries = [w for w in word.vocab if not
               (word.is_oov or word.is_punct or word.like_num or
                word.is_stop or word.lower_ == "rt")
               and w.has_vector and w.lower_ != word.lower_
               and w.is_lower == word.is_lower and w.prob >= -15]

    by_similarity = sorted(
        queries, key=lambda w: word.similarity(w), reverse=True)
    cosine_score = [word.similarity(w) for w in by_similarity]
    return by_similarity[:k], cosine_score[:k]


def select_candidate_codewords(**kwargs):
    """ Select words that share a similarity (functional) or relatedness with a known
    hate speech word. Similarity and relatedness are depenedent on the model passed.
    The idea is to trim the passed vocab.
    Args
    ----

        biased_embeddings (list): Stores (gensim.models): Gensim word embedding model

            0: dep2vec_embedding
            1: word_embedding (either fasttext or w2v)

        unbiased_embeddings (list): Stores (gensim.models): Gensim word embedding model
            0: base_dep2vec_embedding
            1: base_word_embedding (either fasttext or w2v)

        freq_vocab_pair (list):
            0: biased_vocab_freq (dict): Dictionary of token:frequency values. Calculated
                             on the number of documents where that token appears.
            1: unbiased_vocab_freq (dict)

        idf_vocab_pair (list):
            0: biased_vocab_idf (dict): Dictionary of token:idf values. Calculated
                             on the number of documents where that token appears.
            1: unbiased_vocab_idf (dict)

        hs_keywords (set)
        hs_check (bool) : Check for hs keywords or not.

        topn (int): Number of words to check against in the embedding model. default=5.

        p_at_k_threshold (float): Threshold for P@k calculation on the number of
                HS keyword matches that need to appear in the topn, default=0.2.

    Returns:
         dict: dictionary of token:probabiliy pairs.
    """

    dep2vec_embedding = kwargs["biased_embeddings"][0]

    biased_vocab_freq = kwargs["freq_vocab_pair"][0]
    unbiased_vocab_freq = kwargs["freq_vocab_pair"][1]

    kwargs["topn"] = 5 if "topn" not in kwargs else kwargs["topn"]
    kwargs["p_at_k_threshold"] = 0.2 if "p_at_k_threshold" not in kwargs else kwargs[
        "p_at_k_threshold"]

    candidate_codewords = {}
    dep2vec_embedding_vocab = set(dep2vec_embedding.index2word)
    for token in biased_vocab_freq:
        if token in dep2vec_embedding_vocab:
            contextual_representation = get_contextual_representation(
                token=token, biased_embeddings=kwargs["biased_embeddings"],
                unbiased_embeddings=kwargs["unbiased_embeddings"], topn=kwargs["topn"])

            if kwargs["hs_check"]:
                similar_word_set = set(
                    [entry[0] for entry in contextual_representation["similar_words"]])

                # if contextual_representation["similar_words_unbiased"]:
                #     similar_word_set = similar_word_set.union(
                # set([entry[0] for entry in
                # contextual_representation["similar_words_unbiased"]]))

                if contextual_representation["related_words"]:
                    related_word_set = set(
                        [entry[0] for entry in contextual_representation["related_words"]])

                    # if contextual_representation["related_words_unbiased"]:
                    #     related_word_set = related_word_set.union(
                    # set([entry[0] for entry in
                    # contextual_representation["related_words_unbiased"]]))

                sim_intersection = kwargs[
                    "hs_keywords"].intersection(similar_word_set)

                rel_intersection = kwargs[
                    "hs_keywords"].intersection(related_word_set)

                p_at_k_sim = round(float(len(sim_intersection)) /
                                   float(kwargs["topn"]), 3)

                p_at_k_rel = round(float(len(rel_intersection)) /
                                   float(kwargs["topn"]), 3)

                freq_compare = unbiased_vocab_freq[
                    token] if token in unbiased_vocab_freq else 0

                # TODO Need to refine this
                if (p_at_k_sim >= kwargs["p_at_k_threshold"] or p_at_k_rel >=
                        kwargs["p_at_k_threshold"]) and biased_vocab_freq[token] > freq_compare:

                    settings.logger.debug(
                        "Token: %s | Set: %s", token, similar_word_set)

                    candidate_codewords[token] = prep_code_word_representation(
                        token=token, contextual_representation=contextual_representation,
                        freq_vocab_pair=kwargs[
                            "freq_vocab_pair"], topn=kwargs["topn"],
                        idf_vocab_pair=kwargs["idf_vocab_pair"],
                        hs_keywords=kwargs["hs_keywords"])
            else:
                check_intersection = set(
                    [entry[0] for entry in contextual_representation["similar_words"]])

                diff = kwargs["hs_keywords"].intersection(check_intersection)
                if not diff:
                    candidate_codewords[token] = prep_code_word_representation(
                        token=token, contextual_representation=contextual_representation,
                        freq_vocab_pair=kwargs[
                            "freq_vocab_pair"], topn=kwargs["topn"],
                        idf_vocab_pair=kwargs["idf_vocab_pair"],
                        hs_keywords=kwargs["hs_keywords"])

    return candidate_codewords


def prep_code_word_representation(**kwargs):
    """ Helper function for preparing the dictionary fields.
    Args
    ----
    token

    contextual_representation

    freq_vocab_pair

    idf_vocab_pair

    hs_keywords

    topn
    """

    token = kwargs["token"]
    similar_words = kwargs["contextual_representation"]["similar_words"]
    related_words = kwargs["contextual_representation"]["related_words"]
    similar_words_unbiased = kwargs[
        "contextual_representation"]["similar_words_unbiased"]
    related_words_unbiased = kwargs[
        "contextual_representation"]["related_words_unbiased"]

    hs_keywords = kwargs["hs_keywords"]
    topn = kwargs["topn"]

    biased_vocab_freq = kwargs["freq_vocab_pair"][0]
    unbiased_vocab_freq = kwargs["freq_vocab_pair"][1]

    biased_vocab_idf = kwargs["idf_vocab_pair"][0]
    unbiased_vocab_idf = kwargs["idf_vocab_pair"][1]

    hs_similar_words = [
        word for word in similar_words if word[0] in hs_keywords]

    other_similar_words = [
        word for word in similar_words if word[0] not in hs_keywords]

    hs_related_words = [word for word in related_words if word[
        0] in hs_keywords] if related_words else None

    other_related_words = [word for word in related_words if word[
        0] not in hs_keywords] if related_words else None

    hs_similar_words_unbiased = [word for word in similar_words_unbiased if word[
        0] in hs_keywords] if similar_words_unbiased else None

    other_similar_words_unbiased = [word for word in similar_words_unbiased if word[
        0] not in hs_keywords] if similar_words_unbiased else None

    hs_related_words_unbiased = [word for word in related_words_unbiased if word[
        0] in hs_keywords] if related_words_unbiased else None

    other_related_words_unbiased = [word for word in related_words_unbiased if word[
        0] not in hs_keywords] if related_words_unbiased else None

    data = {"freq_biased": [biased_vocab_freq[token]] if token in biased_vocab_freq else [0],
            "freq_unbiased": [unbiased_vocab_freq[token]] if token in unbiased_vocab_freq else [0],

            "idf_biased": [biased_vocab_idf[token]] if token in biased_vocab_idf else [0],
            "idf_unbiased": [unbiased_vocab_idf[token]] if token in unbiased_vocab_idf else [0],

            "hs_supp_sim": compute_avg_cosine(hs_similar_words, topn),
            "hs_supp_rel": compute_avg_cosine(hs_related_words, topn) if hs_related_words else [0],

            "p@k_sim_biased": [round(float(len(hs_similar_words)) /
                                     float(topn), 3)] if hs_similar_words else [0],
            "p@k_rel_biased": [round(float(len(hs_related_words)) /
                                     float(topn), 3)] if hs_related_words else [0],

            "p@k_sim_unbiased": [round(float(len(hs_similar_words_unbiased)) /
                                       float(topn), 3)] if hs_similar_words_unbiased else [0],
            "p@k_rel_unbiased": [round(float(len(hs_related_words_unbiased)) /
                                       float(topn), 3)] if hs_related_words_unbiased else [0],

            "sim_words_hs_biased": [word[0] for word in hs_similar_words
                                    ] if hs_similar_words else [],

            "sim_words_alt_biased": [word[0] for word in other_similar_words
                                     ] if other_similar_words else [],

            "rel_words_hs_biased": [word[0] for word in hs_related_words
                                    ] if hs_related_words else [],

            "rel_words_alt_biased": [word[0] for word in other_related_words
                                     ] if other_related_words else [],

            "sim_words_hs_unbiased": [word[0] for word in hs_similar_words_unbiased
                                      ] if hs_similar_words_unbiased else [],

            "sim_words_alt_unbiased": [word[0] for word in other_similar_words_unbiased
                                       ] if other_similar_words_unbiased else [],

            "rel_words_hs_unbiased": [word[0] for word in hs_related_words_unbiased
                                      ] if hs_related_words_unbiased else [],

            "rel_words_alt_unbiased": [word[0] for word in other_related_words_unbiased
                                       ] if other_related_words_unbiased else []}
    return data


def compute_avg_cosine(similarity_result, topn):
    """ Compute and return the average cosine similarities.
    Args
    ----

        similarity_result (list): List of word:cosine similarity dict pairs.
    Return:
        avg_cosine (float): Computed average cosine similarity
    """
    if len(similarity_result) >= 1:
        cosine_vals = [cos[1] for cos in similarity_result]
        avg_cosine = sum(cosine_vals) / float(topn)
    else:
        return [0]
    return [avg_cosine]


def get_contextual_representation(**kwargs):
    """ Given a word we attempt to create a contextual representation that consists of
    semantically related words, semantically similar words and the avg word vector.
    We compute this from both a biased an unbiased embedding. By bias we refer to the
    method in which the data was originally collected, unbiased means that we streamed from
    Twitter with no keyword arguments.
    Args
    ----

        word (string): Target word.

        biased_embeddings (list): Stores (gensim.models): Gensim word embedding model
            0: dep2vec_embedding
            1: word_embedding (either fasttext or w2v)

        unbiased_embeddings (list): Stores (gensim.models): Gensim word embedding model
            0: base_dep2vec_embedding
            1: base_word_embedding (either fasttext or w2v)

        topn (int): Number of words to check against in the embedding model, default=5.
    """

    word = kwargs["token"]
    topn = 5 if "topn" not in kwargs else kwargs["topn"]

    dep2vec_embedding = kwargs["biased_embeddings"][0]
    word_embedding = kwargs["biased_embeddings"][1]

    base_dep2vec_embedding = kwargs["unbiased_embeddings"][0]
    base_word_embedding = kwargs["unbiased_embeddings"][1]

    contextual_representation = {}
    contextual_representation["similar_words"] = dep2vec_embedding.similar_by_word(
        word, topn=topn, restrict_vocab=None)

    contextual_representation["related_words"] = word_embedding.similar_by_word(
        word, topn=topn, restrict_vocab=None) if word in word_embedding.index2word else None

    contextual_representation["similar_words_unbiased"] = base_dep2vec_embedding.similar_by_word(
        word, topn=topn, restrict_vocab=None) if word in base_dep2vec_embedding.index2word else None

    contextual_representation["related_words_unbiased"] = base_word_embedding.similar_by_word(
        word, topn=topn, restrict_vocab=None) if word in base_word_embedding.index2word else None

    return contextual_representation


def get_average_word_vector(model, word_list):
    """ Average all of the word vectors in a given word list.
    Args:
        model (gensim.models): Gensim word embedding model.
        word_list (list): List of words to average.
    """

    model_vocab = set(model.index2word)
    dimensions = model.vector_size
    feature_vec = np.zeros((dimensions,), dtype="float32")

    for word in word_list:
        if word in model_vocab:
            feature_vec = np.add(feature_vec, model[word])

    feature_vec = np.divide(feature_vec, len(word_list))
    return feature_vec
