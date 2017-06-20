# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various functions that are used to enrich the
contextual representation of words.
"""

import numpy as np
from . import settings


def gensim_top_k_similar(model, row, field_name, k):
    """ Returns the top k similar word vectors from a word embedding model.
    Args:
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
    Args:
        word (spacy.token): Gensim word embedding model.
        k (int): Number of results to return.
    """
    queries = [w for w in word.vocab if not (word.is_oov or word.is_punct or word.like_num or word.is_stop or word.lower_ == "rt")
               and w.has_vector and w.lower_ != word.lower_ and w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(
        queries, key=lambda w: word.similarity(w), reverse=True)
    cosine_score = [word.similarity(w) for w in by_similarity]
    return by_similarity[:k], cosine_score[:k]


def select_candidate_codewords(dep2vec_model, word_embedding_model, vocab, hs_keywords, topn=5, hs_threshold=1):
    """ Select words that share a similarity (functional) or relatedness with a known
    hate speech word. Similarity and relatedness are depenedent on the model passed.
    The idea is to trim the passed vocab.
    Args:

        model (gensim.models): Gensim word embedding model.
        vocab  (dict): Dictionary of token:probability values. Probability is calculated
        on the number of documents where that token appears.
        topn (int): Number of words to check against in the embedding model.
        hs_threshold (int): Number of HS keyword matches that need to appear in the topn results.
    Returns:
         dict: dictionary of token:probabiliy pairs.
    """

    candidate_codewords = {}
    dep2vec_model_vocab = set(dep2vec_model.index2word)
    for token in vocab:
        if token in dep2vec_model_vocab:
            contextual_representation = get_contextual_representation(
                token, dep2vec_model_vocab, word_embedding_model, topn=5)

            check_intersection = set(
                [entry[0] for entry in contextual_representation["similar_words"]])

            diff = hs_keywords.intersection(check_intersection)
            if len(diff) >= hs_threshold:
                hs_similar_words = [
                    word for word in contextual_representation["similar_words"] if word[0] in hs_keywords]

                other_similar_words = [
                    word for word in contextual_representation["similar_words"] if word[0] not in hs_keywords]

                hs_related_words = [
                    word for word in contextual_representation["related_words"] if word[0] in hs_keywords]

                other_related_words = [
                    word for word in contextual_representation["related_words"] if word[0] not in hs_keywords]

                settings.logger.debug("Token: %s | Set: %s", token, diff)

                candidate_codewords[token] = {"probability": vocab[token],
                                              "similar_hs_support": compute_avg_cosine(hs_similar_words),
                                              "hs_similar_words": [word[0] for word in hs_similar_words],
                                              "other_similar_words": [word[0] for word in other_similar_words],
                                              "related_hs_support": compute_avg_cosine(hs_related_words),
                                              "hs_related_words": [word[0] for word in hs_related_words],
                                              "other_related_words": [word[0] for word in other_related_words]}

    return candidate_codewords


def compute_avg_cosine(similarity_result):
    """ Compute and return the average cosine similarities.
    Args:
        similarity_result (list): List of word:cosine similarity dict pairs.
    Return:
        avg_cosine (float): Computed average cosine similarity
    """
    cosine_vals = [cos[1] for cos in similarity_result]
    avg_cosine = sum(cosine_vals) / len(cosine_vals)
    return avg_cosine


def get_contextual_representation(word, dep2vec_model, word_embedding_model, topn=5):
    """ Given a word we attempt to create a contextual representation that consists of
    semantically related words, semantically similar words and the avg word vector.
    Args:
        word (string): Target word.
        dep2vec_model (gensim.models): Gensim word embedding model trained with dep2vec.
        word_embedding_model (gensim.models): Gensim word embedding model trained with
        either fasttext or w2v.
        topn (int): Number of words to check against in the embedding model.
    """
    contextual_representation = {}
    contextual_representation["similar_words"] = dep2vec_model.similar_by_word(
        word, topn=topn, restrict_vocab=None)
    contextual_representation["related_words"] = word_embedding_model.similar_by_word(
        word, topn=topn, restrict_vocab=None)

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
