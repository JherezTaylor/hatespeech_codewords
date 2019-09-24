#! /usr/bin/env python
# -*- coding: utf-8
'''
Python implementation of Krippendorff's alpha -- inter-rater reliability

(c)2011-17 Thomas Grill (http://grrrr.org)

Python version >= 2.4 required
'''

from __future__ import print_function
try:
    import numpy as np
except ImportError:
    np = None


class KrippendorffAlpha:

    def __init__(self, frequencies):
        self.frequencies = frequencies

    def nominal_metric(self, a, b):
        """For nominal data"""
        return a != b

    def interval_metric(self, a, b):
        """For interval data"""
        return (a - b)**2

    def ratio_metric(self, a, b):
        """For ratio data"""
        return ((a - b) / (a + b))**2

    def ordinal_metric(self, a, b):
        """For ordinal data
            frequencies (Collections.Counter): stores the frequencies of ratings
        """
        return (self.frequencies[a] + self.frequencies[b] - (self.frequencies[a] + self.frequencies[b]) / 2) ** 2

    def krippendorff_alpha(self, data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
        '''
        Calculate Krippendorff's alpha (inter-rater reliability):

        data is in the format
        [
            {unit1:value, unit2:value, ...},  # coder 1
            {unit1:value, unit3:value, ...},   # coder 2
            ...                            # more coders
        ]
        or it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, 
        e.g.) with rows corresponding to coders and columns to items

        metric: function calculating the pairwise distance
        force_vecmath: force vector math for custom metrics (numpy required)
        convert_items: function for the type conversion of items (default: float)
        missing_items: indicator for missing items (default: None)
        '''

        # number of coders
        _m = len(data)

        # set of constants identifying missing values
        maskitems = list(missing_items)
        if np is not None:
            maskitems.append(np.ma.masked_singleton)

        # convert input data to a dict of items
        units = {}
        for _d in data:
            try:
                # try if d behaves as a dict
                diter = _d.items()
            except AttributeError:
                # sequence assumed for d
                diter = enumerate(_d)

            for it, g in diter:
                if g not in maskitems:
                    try:
                        its = units[it]
                    except KeyError:
                        its = []
                        units[it] = its
                    its.append(convert_items(g))

        units = dict((it, d) for it, d in units.items()
                     if len(d) > 1)  # units with pairable values
        n = sum(len(pv) for pv in units.values())  # number of pairable values

        if n == 0:
            raise ValueError("No items to compare.")

        np_metric = (np is not None) and (
            (metric in (self.interval_metric, self.nominal_metric, self.ratio_metric)) or force_vecmath)

        Do = 0.
        for grades in units.values():
            if np_metric:
                gr = np.asarray(grades)
                Du = sum(np.sum(metric(gr, gri)) for gri in gr)
            else:
                Du = sum(metric(gi, gj) for gi in grades for gj in grades)
            Do += Du / float(len(grades) - 1)
        Do /= float(n)

        De = 0.
        for g1 in units.values():
            if np_metric:
                d1 = np.asarray(g1)
                for g2 in units.values():
                    De += sum(np.sum(metric(d1, gj)) for gj in g2)
            else:
                for g2 in units.values():
                    De += sum(metric(gi, gj) for gi in g1 for gj in g2)
        De /= float(n * (n - 1))

        return 1. - Do / De if (Do and De) else 1.


if __name__ == '__main__':
    print("Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha")

    data = (
        "*    *    *    *    *    3    4    1    2    1    1    3    3    *    3",  # coder A
        "1    *    2    1    3    3    4    3    *    *    *    *    *    *    *",  # coder B
        "*    *    2    1    3    4    4    *    2    1    1    3    3    *    4",  # coder C
    )

    # missing = '*'  # indicator for missing values
    # # convert to 2D list of string items
    # array = [d.split() for d in data]

    # print("nominal metric: %.3f" % krippendorff_alpha(
    #     array, nominal_metric, missing_items=missing))
    # print("interval metric: %.3f" % krippendorff_alpha(
    #     array, interval_metric, missing_items=missing))
