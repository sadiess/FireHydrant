#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
from sklearn import metrics


def rmseff(x, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    x = np.sort(x, kind="mergesort")
    m = int(c * len(x)) + 1
    return np.min(x_sorted[m:] - x_sorted[:-m]) / 2.0


class ROCPlot:
    def __init__(
        self,
        logscale=False,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        rlim=None,
        height_ratios=[2, 1],
        percent=False,
        grid=False,
        ncol=1,
    ):

        self.gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)

        self.axis = plt.subplot(self.gs[0])
        self.axr = plt.subplot(self.gs[1])

        self.gs.update(wspace=0.025, hspace=0.075)
        plt.setp(self.axis.get_xticklabels(), visible=False)

        if xlim is None:
            xlim = (0.0, 1.0)

        self._xlim = xlim
        self._ylim = ylim

        if xlabel is None:
            xlabel = "True positive rate"
        if ylabel is None:
            ylabel = "False positive rate"

        if percent:
            xlabel = xlabel + " [%]"
            ylabel = ylabel + " [%]"

        self._logscale = logscale
        self._percent = percent
        self._scale = 1 + 99 * percent

        self.axis.set_ylabel(ylabel)
        self.axr.set_xlabel(xlabel)
        self.axr.set_ylabel("Ratio")

        self.axis.grid(grid)
        self.axr.grid(grid)

        self.axis.set_xlim([x * self._scale for x in xlim])
        self.axr.set_xlim([x * self._scale for x in xlim])
        if not ylim is None:
            self.axis.set_ylim([y * self._scale for y in ylim])
        if not rlim is None:
            self.axr.set_ylim(rlim)

        self.auc = []

        self._ncol = ncol

    def plot(self, y_true, y_score, **kwargs):

        fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

        self.auc.append(metrics.roc_auc_score(y_true, y_score))

        if not hasattr(self, "fpr_ref"):
            self.fpr_ref = fpr
            self.tpr_ref = tpr

        sel = tpr >= self._xlim[0]
        if self._logscale:
            self.axis.semilogy(tpr[sel] * self._scale, fpr[sel] * self._scale, **kwargs)
        else:
            self.axis.plot(tpr[sel] * self._scale, fpr[sel] * self._scale, **kwargs)
        r = fpr / np.interp(tpr, self.tpr_ref, self.fpr_ref)
        self.axr.plot(tpr[sel] * self._scale, r[sel], **kwargs)

        self.axis.legend(loc="upper left", ncol=self._ncol)
        if self._percent:
            #self.axis.get_yaxis().set_major_formatter(FormatStrFormatter("%.0f"))
            self.axis.get_yaxis().set_major_formatter(PercentFormatter(decimals=1, symbol=None))
