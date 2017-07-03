# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses functions for building word graphs and
computing results on them.
"""

from math import log10
from collections import Counter
import networkx as nx
from . import text_preprocessing


def init_digraph():
    """ Initialize an empty directed graph"""
    return nx.DiGraph()


def compose_graphs(G, H):
    """ Return a new graph of G composed with H.
    Args
    ----

        G,H (networkX graph)
        Returns:
            C: A new graph with the same type as G.
    """
    return nx.compose(G, H)


def compute_pagerank(G, alpha=0.85):
    """Return the PageRank of the nodes in the graph

    Returns:
        pagerank: (dict) of nodes with PageRank as value.
    """
    return nx.pagerank_scipy(G, alpha=alpha)


def get_leaf_nodes(G):
    """Return a list of nodes in the graph that do not have children."""
    leafs = [node for node in G.nodes() if not G.successors(node)]
    return leafs


def get_parent_nodes(G):
    """Return a list of nodes in the graph that have children"""
    parents = [node for node in G.nodes() if G.successors(node)]
    return parents


def build_wordlist_dg(**kwargs):
    """ Accept a target_word_list and builds a directed graph based on
    the results returned by model.similar_by_word. Weights are initialized
    to 1. For each word in target_word_list we call build_word_directed_graph and merge the results.
    The idea is to build a similarity graph that increases the weight of an edge each
    time a node appears in the similarity results.
    Args
    ----

        wordlist (list): List of words that will act as nodes.

        model (gensim.models): Gensim word embedding model.

        depth (int): Depth to restrict the search to.

        hs_keywords (set)

        topn (int): Number of words to check against in the embedding model, default=5.
    """

    wordlist = kwargs["wordlist"]
    model = kwargs["model"]
    depth = kwargs["depth"]
    hs_keywords = kwargs["hs_keywords"]
    topn = 5 if "topn" not in kwargs else kwargs["topn"]

    wordlist_graph = init_digraph()
    model_vocab = set(model.index2word)

    # Changing how we boost from HS keywords
    single_plural_boost = set()
    hs_checker = (word for word in hs_keywords if word in model_vocab)
    for word in hs_checker:
        single_plural = text_preprocessing.singles_plurals(word)
        for entry in single_plural:
            if entry in model_vocab:
                single_plural_boost.add(entry)

    hs_keywords = hs_keywords.union(single_plural_boost)
    hs_checker = (word for word in hs_keywords if word in model_vocab)

    boost_check = []
    for hs_word in hs_checker:
        boost_check.extend(
            [word[0] for word in model.similar_by_word(hs_word, topn=20)])

    boost_counter = Counter()
    for word in boost_check:
        boost_counter[word] += 1

    for target_word in wordlist:
        # do_hs_boosting = (
        # hs_keywords and model_vocab and target_word in hs_keywords and
        # target_word in model_vocab)
        do_hs_boosting = (
            hs_keywords and model_vocab and target_word in model_vocab)
        if do_hs_boosting:
            target_word_graph = build_word_dg(
                target_word, model, depth, model_vocab=model_vocab, topn=topn, boost_counter=boost_counter)
            wordlist_graph = nx.compose(wordlist_graph, target_word_graph)

        elif not do_hs_boosting and target_word in model_vocab:
            target_word_graph = build_word_dg(
                target_word, model, depth, topn=topn)
            wordlist_graph = nx.compose(wordlist_graph, target_word_graph)
    return wordlist_graph


def build_word_dg(target_word, model, depth, model_vocab=None, boost_counter=None, topn=5):
    """ Accept a target_word and builds a directed graph based on
    the results returned by model.similar_by_word. Weights are initialized
    to 1. Starts from the target_word and gets similarity results for it's children
    and so forth, up to the specified depth.
    Args
    ----

        target_word (string): Root node.

        model (gensim.models): Gensim word embedding model.

        depth (int): Depth to restrict the search to.

        topn (int): Number of words to check against in the embedding model, default=5.
    """

    _DG = init_digraph()
    seen_set = set()
    do_hs_boosting = (
        boost_counter and model_vocab and target_word in model_vocab)

    if do_hs_boosting:
        weight_boost = log10(float(model.vocab[target_word].count)) * boost_counter[
            target_word] if target_word in boost_counter else 0
        _DG.add_weighted_edges_from([(target_word, word[0], weight_boost + word[1])
                                     for word in model.similar_by_word(target_word, topn=topn)])
    else:
        _DG.add_weighted_edges_from([(target_word, word[0], word[1])
                                     for word in model.similar_by_word(target_word, topn=topn)])
    seen_set.add(target_word)
    for _idx in range(1, depth):
        current_nodes = _DG.nodes()
        for node in current_nodes:
            if node not in seen_set:
                _DG.add_weighted_edges_from(
                    [(node, word[0], word[1]) for word in model.similar_by_word(node, topn=topn)])
                seen_set.add(node)
    return _DG


def build_converged_graph(**kwargs):
    """ The idea here is to build a graph from a given list and expand the graph
    until it converges or until a specified number of rounds. On each iteration we store
    the nodes that were already seen and repeat the process with the graph node difference.
    Args
    ----


        wordlist (list): List of words that will act as nodes.

        model (gensim.models): Gensim word embedding model.

        depth (int): Depth to restrict the search to.

        rounds (int): Number of times to repeat the process.

        hs_keywords (set)

        topn (int): Number of words to check against in the embedding model, default=5.
    """

    rounds = kwargs["rounds"]
    candidate_graph = init_digraph()
    for _idx in range(0, rounds):
        graph = build_wordlist_dg(**kwargs)
        candidate_graph = compose_graphs(candidate_graph, graph)
        leafs = get_leaf_nodes(candidate_graph)
        kwargs["wordlist"] = list(leafs)
    return candidate_graph
