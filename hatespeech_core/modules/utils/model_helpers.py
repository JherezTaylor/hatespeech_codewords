# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses various helper functions for use with the various models
"""

import joblib
import gensim
import fasttext
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import IncrementalPCA
from sklearn.base import TransformerMixin, BaseEstimator
import cufflinks as cf
import plotly as py
import plotly.graph_objs as go
from . import settings
from . import notifiers
from ..db import mongo_base


def init_plotly():
    """ Initialize plot.ly in offline mode
    """
    py.offline.init_notebook_mode(connected=True)
    cf.set_config_file(offline=True)


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


def plot_confusion_matrix(_cm, labels, chart_title, filename):
    """ Accept and plot a confusion matrix
    Args:
        cm (numpy.array): Confusion matrix
        labels (list): Feature labels
        chart_title (str)
        filename (str)
    """

    labels = sorted(labels.tolist())
    trace = go.Heatmap(z=_cm, x=labels, y=labels,
                       colorscale='Blues', reversescale=True)
    data = [trace]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title, xaxis=dict(title='Predicted Label'),
                            yaxis=dict(title='True Label',
                                       autorange='reversed')
                            )
    })


def plot_scatter_chart(values, chart_title):
    """ Accept and plot values on a scatter chart
    Args:
        value_list  (list): Numerical values to plot.
        chart_title (str)
        filename (str)
    """

    trace = go.Scatter(
        x=list(range(1, len(values) + 1)),
        y=values
    )
    data = [trace]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title)
    })


def plot_histogram(values):
    """ Accept and plot values on a histogram
    Args:
       value_list  (list): Numerical values to plot.
       chart_title (str)
       filename (str)
    """
    data = [go.Histogram(x=values)]
    py.offline.iplot({
        "data": data
    })


def plot_word_embedding(X_tsne, vocab, chart_title, show_labels):
    """ Accept and plot values for a word embedding
    Args:
       value_list  (list): Numerical values to plot.
       chart_title (str)
       filename (str)
    """
    if show_labels:
        display_mode = 'markers+text'
        display_text = vocab
    else:
        display_mode = 'markers'
        display_text = None

    trace = go.Scatter(
        x=X_tsne[:, 0],
        y=X_tsne[:, 1],
        mode=display_mode,
        text=display_text,
        marker=dict(size=14,
                    line=dict(width=0.5),
                    opacity=0.3,
                    color='rgba(217, 217, 217, 0.14)'
                    )
    )
    data = [trace]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title)
    })


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
            plot_confusion_matrix(
                cm, y.unique(), process_name, process_name + "_cm")
#     print(sum(scores) / num_expts)


def pca_reduction(vectors, num_dimensions, model_name):
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


def gensim_top_k_similar(model, row, field_name, k):
    """ Returns the top k similar word vectors from a word embedding model.
    Args:
        model (gensim.models): Gensim word embedding model.
        row (pandas.Dataframe): Row extracted from a dataframe.
        field_name (str): Name of dataframe column to extract.
        k (int): Number of results to return.
    """

    similar_words = []
    for word in row[field_name]:
        if word in model.vocab:
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


@notifiers.do_cprofile
def train_fasttext_model(input_data, filename):
    """ Train a fasttext model
    """
    cpu_count = joblib.cpu_count()
    _model = fasttext.skipgram(input_data, filename, thread=cpu_count)


@notifiers.do_cprofile
def train_word2vec_model(input_data, filename):
    """ Train a word2vec model
    """
    cpu_count = joblib.cpu_count()
    sentences = gensim.models.word2vec.LineSentence(input_data)
    model = gensim.models.Word2Vec(sentences, min_count=5, workers=cpu_count)
    model.save(filename)


# @notifiers.do_cprofile
def train_fasttext_classifier(input_data, filename, lr=0.1, dim=100, ws=5, epoch=5, min_count=1, word_ngrams=1):
    """ Train a fasttext model
    See https://github.com/salestock/fastText.py for params. 
    """
    cpu_count = joblib.cpu_count()
    _classifier = fasttext.supervised(
        input_data, filename, thread=cpu_count, lr=lr, dim=dim, ws=5, epoch=epoch, min_count=1, word_ngrams=1)
