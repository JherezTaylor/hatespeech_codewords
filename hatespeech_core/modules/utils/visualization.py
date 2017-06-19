# Author: Jherez Taylor <jherez.taylor@gmail.com>
# License: MIT
# Python 3.5

"""
This module houses plotting and visualization functions.
"""

import cufflinks as cf
import plotly as py
import plotly.graph_objs as go


def init_plotly():
    """ Initialize plot.ly in offline mode
    """
    py.offline.init_notebook_mode(connected=True)
    cf.set_config_file(offline=True)

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

    trace = go.Scattergl(
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