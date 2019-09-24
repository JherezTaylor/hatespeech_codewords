# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various helper functions for use with the various models
"""

from math import log10
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import IncrementalPCA
from sklearn.base import TransformerMixin, BaseEstimator
from . import file_ops
from . import settings
from . import visualization
from ..db import mongo_base


def fetch_as_df(connection_params, projection):
    """ Takes MongoDB connection params and returns the specified
    collection as a pandas dataframe.

    Args:
        connection_params  (list): Contains connection objects and params as follows:
            0: db_name     (str): Name of database to query.
            1: collection  (str): Name of collection to use.
        projection (dict): Dictionary of fields to return, returns all fields if blank.
    """

    client = mongo_base.connect()
    connection_params.insert(0, client)
    query = {}
    query["filter"] = {}
    query["projection"] = projection
    query["limit"] = 0
    query["skip"] = 0
    query["no_cursor_timeout"] = True
    cursor = mongo_base.finder(connection_params, query, False)
    _df = pd.DataFrame(list(cursor))
    return _df


def run_experiment(X, y, pipeline, process_name, display_args, num_expts=1):
    """ Accept features, labels and a model pipeline for running an underlying classifier model
    Args:
        X (pandas.Dataframe): Model features
        y (pandas.Series): Feature labels.
        pipeline (sklearn.pipeline): Underlying classifier model.
        process_name (str): Model description.
        display_args (list):
            0: Boolean for printing confusion matrix.
            1: Boolean for plotting confusion matrix.
        num_expts (int): Number of times to run model.
        plot_cm=False, print_cm=False
    """
    settings.logger.info("Predicting the labels of the test set...")
    settings.logger.debug("%s documents", len(X))
    settings.logger.debug("%s categories", len(y.value_counts()))

    scores = list()
    for i in tqdm(range(num_expts)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, train_size=0.80)
        model = pipeline.fit(X_train, y_train)  # train the classifier

        # apply the model to the test data
        y_prediction = model.predict(X_test)
        report = classification_report(y_test, y_prediction)

        # compare the results to the gold standard
        score = accuracy_score(y_prediction, y_test)
        scores.append(score)
        settings.logger.info("Classification Report: %s", process_name)
        print(report)

        cm = confusion_matrix(y_test, y_prediction)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(decimals=3)
        if display_args[0]:
            print("Confusion matrix: ")
            print(cm)
        if display_args[1]:
            visualization.plot_confusion_matrix(
                cm, y.unique(), process_name, process_name + "_cm")
#     print(sum(scores) / num_expts)


def pca_reduction(vectors, num_dimensions, model_name):
    """ Run dimensionality reduction using IncrementalPCA
    Args:
        vectors: Word vectors to be reduced.
        num_dimensions (int): Number of dimensions to reduce to.
        model_name (string)
    """

    settings.logger.info(
        "Reducing to %sD using IncrementalPCA...", num_dimensions)
    ipca = IncrementalPCA(n_components=num_dimensions)
    vectors = ipca.fit_transform(vectors)
    joblib.dump(vectors, model_name, compress=True)
    settings.logger.info("Reduction Complete")
    return vectors


def run_gridsearch_cv(pipeline, X, y, param_grid, n_jobs, score):
    """ Perfrom  k-fold grid search across the data in order to fine tune
    the parameters.
    Args:
        pipeline (sklearn.pipeline): Underlying classifier model.
        X (pandas.Dataframe): Model features
        y (pandas.Series): Feature labels.
        param_grid (list): List of hyperparameters to validate.
        n_jobs (int): Number of threads to utilize.
        score (string): Scoring mechanism to utilize [recall or precision].
    """

    print("Tuning hyper-parameters for {0}\n".format(score))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    clf = GridSearchCV(pipeline, param_grid=param_grid, cv=2,
                       n_jobs=4, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("Grid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r\n" % (mean, std * 2, params))

    print("Detailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))


def evaluate_prediction(predictions, target):
    print(classification_report(target, predictions))
    print("Accuracy: ", accuracy_score(target, predictions))


def get_feature_stats(vectorizer, X, skb, feature_names):
    """ Returns the number of features after vectorization,
    also returns the top features as determined by a chi square test.
    Args:
        vectorizer (sklearn vectorizer)
        X (pandas.Dataframe): Raw features.
        skb (sklearn.selectKBest)
        feature_names (Boolean): Optional, returns top features.
    """

    vectorizer.fit(X)
    if feature_names:
        return len(vectorizer.get_feature_names()), [feature_names[i] for i in skb.get_support(indices=True)]
    else:
        return len(vectorizer.get_feature_names())


def empty_analyzer():
    """ Lambda for use with the following class.
    """

    analyzer = lambda x: x
    return analyzer


class TextExtractor(BaseEstimator, TransformerMixin):
    """ Adapted from code by @zacstewart
    https://github.com/zacstewart/kaggle_seeclickfix/blob/master/estimator.py
    Also see Zac Stewart's excellent blogpost on pipelines:
    http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        """ Select the relevant column and return it as a numpy array
        """
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(str)

    def fit(self, *_):
        return self


class TextListExtractor(BaseEstimator, TransformerMixin):
    """ Extract a list of values from a dataframe.
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        return df[self.column_name].tolist()

    def fit(self, *_):
        return self


class Apply(BaseEstimator, TransformerMixin):
    """Applies a function f element-wise to the numpy array
    """

    def __init__(self, fn):
        self.fn = np.vectorize(fn)

    def transform(self, data):
        # note: reshaping is necessary because otherwise sklearn
        # interprets 1-d array as a single sample
        return self.fn(data.reshape(data.size, 1))

    def fit(self, *_):
        return self


class BooleanExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, column_name):
        self.column_name = column_name

    def transform(self, df):
        # select the relevant column and return it as a numpy array
        # set the array type to be string
        return np.asarray(df[self.column_name]).astype(np.int)

    def fit(self, *_):
        return self


def get_els_word_weights(vocab, total_doc_count, hs_keywords):
    """ Calculate word frequencies and IDF weights from the vocab results returned by
    elasticsearch_base.aggregate()
    Args:
       vocab  (dict): Dictionary of token:doc_count values. doc_count is the number of
                      documents where that token appears.
       total_doc_count (int): Total number of documents in the corpus.
    Returns:
        hs_vocab_freqs (dict): subset of token:frequency pairs for tokens in the HS keywords corpus.
        vocab_freqs (dict): subset of token:frequency pairs for the entire vocab.
        *_idf (dict): token:idf pairs.
        P(wi) = number of docs with (wi) / count(total number of documents)
        IDF = log(total number of documents / number of docs with (wi))
    """

    vocab_frequencies = {}
    hs_vocab_frequencies = {}
    vocab_idf = {}
    hs_vocab_idf = {}

    vocab_frequencies = {token:
                         float(val) / float(total_doc_count) for token, val in vocab.items()}
    vocab_idf = {token: log10(float(total_doc_count) / float(val))
                 for token, val in vocab.items()}

    hs_vocab_frequencies = {token: vocab_frequencies[
        token] for token in vocab_frequencies if token in hs_keywords}
    hs_vocab_idf = {token: vocab_idf[token]
                    for token in vocab_idf if token in hs_keywords}

    return hs_vocab_frequencies, vocab_frequencies, hs_vocab_idf, vocab_idf


def get_overlapping_weights(vocab, comparison_vocab):
    """ Accepts a pair of dictionary that stores token:weight and gets
    the values for tokens that are in both vocabularies.
    Args:
        vocab (dict): token:weight pairs extracted from a corpus.
        comparison_vocab (dict): The vocabulary to be checked.
    Returns:
        token_list_1, token_weight_1, token_list_2, token_weight_2 (list): List of tokens and
        weights resepctively.
    """

    vocab_tokens = set(vocab.keys())
    comparison_tokens = set(comparison_vocab.keys())
    token_intersection = vocab_tokens.intersection(comparison_tokens)

    vocab_list = {}
    comparison_vocab_list = {}

    vocab_list = {token: vocab[token] for token in token_intersection}
    comparison_vocab_list = {token: comparison_vocab[
        token] for token in token_intersection}

    return list(vocab_list.keys()), list(vocab_list.values()), \
        list(comparison_vocab_list.keys()), list(
            comparison_vocab_list.values())


def get_model_vocabulary(model):
    """ Return the stored vocabulary for an embedding model
    Args:
        model (gensim.models) KeyedVectors or Word2Vec model.
    """
    return set(model.vocab.keys())


def get_model_word_count(model, word):
    """ Return the count for a given word in an embedding model
    Args:
        model (gensim.models) KeyedVectors or Word2Vec model
    """
    return model.vocab[word].count
