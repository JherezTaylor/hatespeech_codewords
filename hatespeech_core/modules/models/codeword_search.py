# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module stores functions for selecting candidate codewords.
"""

from joblib import Parallel, delayed, cpu_count
from textblob import TextBlob
from ..utils import settings
from ..utils import file_ops
from ..utils import word_enrichment
from ..utils import graphing


def candidate_codeword_search(**kwargs):
    """ Candidate search process. We define the bulk of the logic in select_candidate_codewords.
    At this level we define the job to be run in parallel and perform the final
    stage of the process, the pagerank results.

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
        graph_depth (int): Depth for the token graph.
        hs_check (bool) : Check for hs keywords or not.

        topn (int): Number of words to check against in the embedding model. default=5.

        p_at_k_threshold (float): Threshold for P@k calculation on the number of
                HS keyword matches that need to appear in the topn, default=0.2.

    Returns:
         dict: dictionary of token:probability pairs.
         dict: pagerank scores.
         candidate_graph
         set: singular_tokens:
    """

    candidate_graph = graphing.init_digraph()
    candidate_codewords = {}
    biased_vocab_freq = kwargs["freq_vocab_pair"][0]

    dep2vec_embedding_vocab = set(kwargs["biased_embeddings"][0].index2word)
    word_embedding_vocab = set(kwargs["biased_embeddings"][1].index2word)

    base_dep2vec_embedding_vocab = set(
        kwargs["unbiased_embeddings"][0].index2word)
    base_word_embedding_vocab = set(
        kwargs["unbiased_embeddings"][1].index2word)

    kwargs["biased_embeddings_vocab_pair"] = [
        dep2vec_embedding_vocab, word_embedding_vocab]
    kwargs["unbiased_embeddings_vocab_pair"] = [
        base_dep2vec_embedding_vocab, base_word_embedding_vocab]

    job_results = []
    num_cores = cpu_count()
    partitions = file_ops.dict_chunks(biased_vocab_freq, num_cores)
    job_results.append(Parallel(n_jobs=num_cores, backend="threading")
                       (delayed(select_candidate_codewords)(
                           partition, **kwargs) for partition in partitions))

    singular_tokens = set()
    for result in job_results:
        for sub_result in result:
            # Requires python 3.5.3 to merge, use file_ops.merge_dicts if fail
            candidate_codewords = {**candidate_codewords, **sub_result["candidate_codewords"]}
            singular_tokens = singular_tokens.union(
                sub_result["singular_intersection"])
            candidate_graph = graphing.compose_graphs(
                candidate_graph, sub_result["candidate_graph"])

    job_results = []
    secondary_pass_tokens = {token: biased_vocab_freq[
        token] for token in singular_tokens if token in biased_vocab_freq}

    kwargs["freq_vocab_pair"][0] = secondary_pass_tokens
    partitions = file_ops.dict_chunks(secondary_pass_tokens, num_cores)
    job_results.append(Parallel(n_jobs=num_cores, backend="threading")
                       (delayed(select_candidate_codewords)(
                           partition, **kwargs) for partition in partitions))

    for result in job_results:
        for sub_result in result:
            candidate_codewords = {**candidate_codewords, **sub_result["candidate_codewords"]}
            singular_tokens = singular_tokens.union(
                sub_result["singular_intersection"])
            candidate_graph = graphing.compose_graphs(
                candidate_graph, sub_result["candidate_graph"])

    # Third codeword condition, trim this list outside the function
    pagerank = graphing.compute_pagerank(candidate_graph, alpha=0.85)
    return candidate_codewords, pagerank, candidate_graph, singular_tokens


def select_candidate_codewords(partition, **kwargs):
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

        biased_embeddings_vocab_pair (list): Stores (gensim.models.index2word):
            0: dep2vec_embedding_vocab
            1: word_embedding_vocab

        unbiased_embeddings_vocab_pair (list): Stores (gensim.models.index2word):
            0: base_dep2vec_embedding_vocab
            1: base_word_embedding_vocab

        freq_vocab_pair (list):
            0: biased_vocab_freq (dict): Dictionary of token:frequency values. Calculated
                             on the number of documents where that token appears.
            1: unbiased_vocab_freq (dict)

        idf_vocab_pair (list):
            0: biased_vocab_idf (dict): Dictionary of token:idf values. Calculated
                             on the number of documents where that token appears.
            1: unbiased_vocab_idf (dict)

        hs_keywords (set)
        graph_depth (int): Depth for the token graph.
        hs_check (bool) : Check for hs keywords or not.

        topn (int): Number of words to check against in the embedding model. default=5.

        p_at_k_threshold (float): Threshold for P@k calculation on the number of
                HS keyword matches that need to appear in the topn, default=0.2.

    Returns:
         dict: dictionary of token:probability pairs.
         set: singular tokens.
         candidate_graph
    """

    dep2vec_embedding = kwargs["biased_embeddings"][0]
    biased_vocab_freq = kwargs["freq_vocab_pair"][0]

    dep2vec_embedding_vocab = kwargs["biased_embeddings_vocab_pair"][0]
    word_embedding_vocab = kwargs["biased_embeddings_vocab_pair"][1]
    base_dep2vec_embedding_vocab = kwargs["unbiased_embeddings_vocab_pair"][0]
    base_word_embedding_vocab = kwargs["unbiased_embeddings"][1]

    kwargs["topn"] = 5 if "topn" not in kwargs else kwargs["topn"]
    kwargs["p_at_k_threshold"] = 0.2 if "p_at_k_threshold" not in kwargs else kwargs[
        "p_at_k_threshold"]

    candidate_graph = graphing.init_digraph()
    candidate_codewords = {}
    singular_words = set()

    for token in partition:
        if kwargs["hs_check"]:
            if token in kwargs["hs_keywords"] or token not in biased_vocab_freq:
                pass

            elif token in dep2vec_embedding_vocab:
                token_graph = graphing.build_word_dg(
                    token, dep2vec_embedding, kwargs["graph_depth"])
                candidate_graph = graphing.compose_graphs(
                    candidate_graph, token_graph)

                contextual_representation = word_enrichment.get_contextual_representation(
                    token=token, biased_embeddings=kwargs["biased_embeddings"],
                    unbiased_embeddings=kwargs[
                        "unbiased_embeddings"], topn=kwargs["topn"],
                    biased_vocab=[dep2vec_embedding_vocab,
                                  word_embedding_vocab],
                    unbiased_vocab=[base_dep2vec_embedding_vocab, base_word_embedding_vocab])

                # Track the singular words for a cleanup pass at the end
                singular_tokens = TextBlob(token).words
                if singular_tokens:
                    for entry in singular_tokens:
                        temp = entry.singularize()
                        if temp and temp != token and temp not in candidate_codewords:
                            singular_words.add(temp)

                # Primary codeword condition. Issue #116
                primary_check = primary_codeword_support(token=token, hs_keywords=kwargs["hs_keywords"],
                                                         contextual_representation=contextual_representation,
                                                         freq_vocab_pair=kwargs[
                                                             "freq_vocab_pair"], topn=kwargs["topn"],
                                                         p_at_k_threshold=kwargs["p_at_k_threshold"])
                if primary_check:
                    secondary_check = secondary_codeword_support(
                        token_graph=token_graph, token=token, hs_keywords=kwargs["hs_keywords"])

                    # Secondary codeword condition. Issue #122
                    if secondary_check[0]:
                        candidate_codewords[token] = word_enrichment.prep_code_word_representation(
                            token=token, contextual_representation=contextual_representation,
                            freq_vocab_pair=kwargs[
                                "freq_vocab_pair"], topn=kwargs["topn"],
                            idf_vocab_pair=kwargs["idf_vocab_pair"],
                            hs_keywords=kwargs[
                                "hs_keywords"], candidate_bucket=["primary"],
                            graph_hs_matches=secondary_check[1]
                        )

                    elif not secondary_check[0]:
                        candidate_codewords[token] = word_enrichment.prep_code_word_representation(
                            token=token, contextual_representation=contextual_representation,
                            freq_vocab_pair=kwargs[
                                "freq_vocab_pair"], topn=kwargs["topn"],
                            idf_vocab_pair=kwargs["idf_vocab_pair"],
                            hs_keywords=kwargs[
                                "hs_keywords"], candidate_bucket=["primary"]
                        )

                # Fails primary condition, try secondary.
                elif not primary_check:
                    secondary_check = secondary_codeword_support(
                        token_graph=token_graph, token=token, hs_keywords=kwargs["hs_keywords"])

                    if secondary_check[0]:
                        candidate_codewords[token] = word_enrichment.prep_code_word_representation(
                            token=token, contextual_representation=contextual_representation,
                            freq_vocab_pair=kwargs[
                                "freq_vocab_pair"], topn=kwargs["topn"],
                            idf_vocab_pair=kwargs["idf_vocab_pair"],
                            hs_keywords=kwargs[
                                "hs_keywords"], candidate_bucket=["secondary"],
                            graph_hs_matches=secondary_check[1]
                        )
        # Don't check for HS keywords
        elif not kwargs["hs_check"]:
            check_intersection = set(
                [entry[0] for entry in contextual_representation["similar_words"]])

            diff = kwargs["hs_keywords"].intersection(check_intersection)
            if not diff:
                candidate_codewords[token] = word_enrichment.prep_code_word_representation(
                    token=token, contextual_representation=contextual_representation,
                    freq_vocab_pair=kwargs[
                        "freq_vocab_pair"], topn=kwargs["topn"],
                    idf_vocab_pair=kwargs["idf_vocab_pair"],
                    hs_keywords=kwargs["hs_keywords"])

    # End main search and do cleanup pass
    # Check for the singular versions of words
    singular_intersection = singular_words.intersection(
        dep2vec_embedding_vocab)

    result = {}
    result["candidate_codewords"] = candidate_codewords
    result["singular_intersection"] = singular_intersection
    result["candidate_graph"] = candidate_graph
    return result


def primary_codeword_support(**kwargs):
    """ Helper function for determining if a candidate word passes
    our first stage requirements for being flagged as a code word.
    Refer to issue #116 https://github.com/JherezTaylor/thesis-preprocessing/issues/116
    Args
    ----

        contextual_representation (dict)

        token (str)

        freq_vocab_pair (list):
            0: biased_vocab_freq (dict): Dictionary of token:frequency values. Calculated
                             on the number of documents where that token appears.
            1: unbiased_vocab_freq (dict)

        hs_keywords (set)

        topn (int): Number of words to check against in the embedding model. default=5.

        p_at_k_threshold (float): Threshold for P@k calculation on the number of
                HS keyword matches that need to appear in the topn, default=0.2.
    Returns:
        boolean: Whether the codeword passes the support requirements.
    """

    contextual_representation = kwargs["contextual_representation"]
    biased_vocab_freq = kwargs["freq_vocab_pair"][0]
    unbiased_vocab_freq = kwargs["freq_vocab_pair"][1]
    token = kwargs["token"]

    similar_word_set = set(
        [entry[0] for entry in contextual_representation["similar_words"] if entry])

    if contextual_representation["related_words"]:
        related_word_set = set(
            [entry[0] for entry in contextual_representation["related_words"] if entry])
    else:
        related_word_set = set()

    sim_intersection = kwargs["hs_keywords"].intersection(similar_word_set)
    rel_intersection = kwargs["hs_keywords"].intersection(related_word_set)

    p_at_k_sim = round(float(len(sim_intersection)) /
                       float(kwargs["topn"]), 3)

    p_at_k_rel = round(float(len(rel_intersection)) /
                       float(kwargs["topn"]), 3)

    freq_compare = unbiased_vocab_freq[
        token] if token in unbiased_vocab_freq else 0

    if (p_at_k_sim >= kwargs["p_at_k_threshold"] or p_at_k_rel >=
            kwargs["p_at_k_threshold"]) and biased_vocab_freq[token] > freq_compare:
        settings.logger.debug(
            "Token: %s | Set: %s", token, similar_word_set)
        return True
    else:
        return False


def secondary_codeword_support(**kwargs):
    """ Helper function for determining if a candidate word passes
    our second stage requirements for being flagged as a code word.
    Refer to issue #122 https://github.com/JherezTaylor/thesis-preprocessing/issues/122
    Args
    ----

    token_graph (networkx.classes.digraph.DiGraph): Directed word similarity
                graph for the given token.

    token (str)

    hs_keywords (set)

    Returns:
        [boolean, graph_hs_matches]: Boolean and a dictionary of node matches and
                            the path from the HS keyword match to its parent.
    """

    graph_hs_matches = {}
    for node in kwargs["token_graph"]:

        singular_tokens = TextBlob(node).words
        if singular_tokens:
            for entry in singular_tokens:
                temp = entry.singularize()
                if temp and temp != kwargs["token"] and (temp in kwargs["hs_keywords"] or node in kwargs["hs_keywords"]):
                    graph_hs_matches[node] = [kwargs[
                        "token_graph"].predecessors(node)]

    if graph_hs_matches:
        return [True, graph_hs_matches]
    else:
        return [False, graph_hs_matches]
