#!/usr/bin/env python
# From https://gist.github.com/neggert/2399228
import numpy as np
import matplotlib.pyplot as plt
import inspect


def hist_errorbars(data, xerrs=True, *args, **kwargs):
    """Plot a histogram with error bars. Accepts any kwarg accepted by either numpy.histogram or pyplot.errorbar"""
    # pop off normed kwarg, since we want to handle it specially
    norm = False
    if "normed" in kwargs.keys():
        norm = kwargs.pop("normed")
    if "density" in kwargs.keys():
        norm = kwargs.pop("density")

    # retrieve the kwargs for numpy.histogram
    histkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.getargspec(np.histogram).args:
            histkwargs[key] = value

    histvals, binedges = np.histogram(data, **histkwargs)
    yerrs = np.sqrt(histvals)

    if norm:
        nevents = float(sum(histvals))
        binwidth = binedges[1] - binedges[0]
        histvals = histvals / nevents / binwidth
        yerrs = yerrs / nevents / binwidth

    bincenters = (binedges[1:] + binedges[:-1]) / 2

    if xerrs:
        xerrs = (binedges[1] - binedges[0]) / 2
    else:
        xerrs = None

    # retrieve the kwargs for errorbar
    ebkwargs = {}
    for key, value in kwargs.items():
        if key in inspect.getfullargspec(plt.errorbar).args:
            histkwargs[key] = value
    out = plt.errorbar(bincenters, histvals, yerrs, xerrs, fmt=".", **ebkwargs)

    if "log" in kwargs.keys():
        if kwargs["log"]:
            plt.yscale("log")

    if "range" in kwargs.keys():
        plt.xlim(*kwargs["range"])

    return out
