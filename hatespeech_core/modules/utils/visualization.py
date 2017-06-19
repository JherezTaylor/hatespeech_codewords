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


def plot_scatter_chart(y_values, chart_title):
    """ Accept and plot values on a scatter chart
    Args:
        y_values  (list): Numerical values to plot.
        chart_title (str)
    """

    trace = go.Scatter(
        x=list(range(1, len(y_values) + 1)),
        y=y_values
    )
    data = [trace]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title)
    })


def plot_histogram(x_values):
    """ Accept and plot values on a histogram
    Args:
       x_values  (list): Numerical values to plot.
       chart_title (str)
       filename (str)
    """

    data = [go.Histogram(x=x_values)]
    py.offline.iplot({
        "data": data
    })


def plot_basic_bar_chart(x_values, y_values, chart_title, orientation="v"):
    """ Accept and plot values on a bar chart
    Args:
       x_values (list): Labels.
       y_values  (list): Numerical values to plot.
       chart_title (str)
       orientation (str): h or v for horizontal or vertical
    """

    if orientation == "h":
        data = [go.Bar(x=y_values, y=x_values, orientation=orientation)]
    else:
        data = [go.Bar(x=x_values, y=y_values, orientation=orientation)]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title)
    })


def plot_bar_chart(x_values, y1_values, y2_values, series1, series2, chart_title, barmode="stack", orientation="v"):
    """ Accept and plot values on a bar chart in either grouped or stacked format.
    Args:
       x_values (list): Labels.
       y1_values  (list): Numerical values to plot.
       y2_values  (list): Numerical values to plot.
       series (str): Name of data series.
       chart_title (str)
       barmode (str): Accepts either stack or group.
    """

    if orientation == "h":
        trace1 = go.Bar(
            x=y1_values,
            y=x_values,
            name=series1,
            orientation=orientation
        )
        trace2 = go.Bar(
            x=y1_values,
            y=x_values,
            name=series2,
            orientation=orientation
        )
    else:
        trace1 = go.Bar(
            x=x_values,
            y=y1_values,
            name=series1,
            orientation=orientation
        )
        trace2 = go.Bar(
            x=x_values,
            y=y2_values,
            name=series2,
            orientation=orientation
        )

    data = [trace1, trace2]
    py.offline.iplot({
        "data": data,
        "layout": go.Layout(title=chart_title, barmode=barmode)
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
