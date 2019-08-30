#!/usr/bin/env python
import concurrent.futures
from collections import MutableMapping, OrderedDict, defaultdict
from functools import reduce
import multiprocessing
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from cycler import cycler
import utils.uprootHelpers as uuh
import utils.commonHelpers as uch

ffPltStyleSettings = {
    "figure.facecolor": "w",
    "figure.figsize": [8.0, 6.0],
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "grid.linestyle": ":",
    "axes.grid": True,
    "axes.prop_cycle": cycler(
        color=[
            "#1696d2",
            "#ec008b",
            "#000000",
            "#d2d2d2",
            "#fdbf11",
            "#55b748",
            "#9e0142",
            "#f46d43",
            "#bf812d",
            "#66c2a5",
            "#5e4fa2",
            "#e6f598",
        ]
    ),
    "axes.titleweight": "semibold",
    "axes.titlesize": "x-large",
    "axes.labelsize": "medium",
    "axes.labelweight": "normal",
    "font.size": 13.0,
    "font.family": ["Lato", "sans-serif"],
    "savefig.bbox": "tight",
}

# ------------------------------------------------------------------------------


def SingleFill(
    histo,
    loaderCls,
    dataKey,
    fillingMethod,
    filename,
    weight=None,
    benchKey=None,
    ax="x",
    weightax="w",
):
    """
    fill histogram with `fillingMethod` for a file with name `filename`.

    :param :py:class:`HistBook.Hist` histo: Hist class
    :param class `loaderCls`: loaderCls with all necessary data attributes stuffed
    :param str dataKey: data attribute name
    :param function fillingMethod: transformation function
    :param str filename: filename to work on
    :param scalar/array-like weight: if not None(default), apply weight
    :param str benchKey: if not None(default), filling profile/efficiency like plots
    :param str ax: histogram's axis name
    :param str weigthax: histogram's weight axis name

    :returns: filled hist
    """

    aLoader = loaderCls(filename)
    aFiller = ffFiller(fillingMethod, aLoader)
    if benchKey is None:
        aFiller.fill(histo, dataKey, ax=ax, weight=weight, weightax=weightax)
    else:
        aFiller.fillp(histo, dataKey, benchKey)

    return histo


# ------------------------------------------------------------------------------


def SingleFill2D(
    histo,
    loaderCls,
    dataKeys,
    fillingMethods,
    filename,
    weight=None,
    benchKey=None,
    ax=["x", "y"],
    weightax="w",
):
    """
    fill 2D histogram with `fillingMethods` for a file with name `filename`.

    :param :py:class:`HistBook.Hist` histo: Hist class
    :param class `loaderCls`: loaderCls with all necessary data attributes stuffed
    :param str dataKeys: data attribute names, len(dataKeys)==2
    :param callable/array-like fillingMethods: transformation function(s)
    :param str filename: filename to work on
    :param scalar/array-like weight: if not None(default), apply weight
    :param str benchKey: if not None(default), filling profile/efficiency like plots
    :param list ax: histogram's axis names, len(ax)==2
    :param str weigthax: histogram's weight axis name

    :returns: filled hist
    """

    aLoader = loaderCls(filename)
    aFiller = ffFiller2D(fillingMethods, aLoader)
    if benchKey is None:
        aFiller.fill(histo, dataKeys, ax=ax, weight=weight, weightax=weightax)
    else:
        aFiller.fillp(histo, dataKeys, benchKey)

    return histo


# ------------------------------------------------------------------------------


def MultiFill(
    histo,
    loaderCls,
    dataKey,
    fillingMethod,
    filelist,
    weight=None,
    benchKey=None,
    ax="x",
    weightax="w",
    strategy=concurrent.futures.ProcessPoolExecutor,
):
    """
    fill histogram with `fillingMethod` for all files in `filelist`.

    :param :py:class:`HistBook.Hist` histo: Hist class
    :param class `loaderCls`: loaderCls with all necessary data attributes stuffed
    :param str datakey: data attribute name
    :param function fillingMethod: transformation function
    :param list filelist: a list of files to work on
    :param scalar/array-like weight: if not None(default), apply weight
    :param str benchKey: if not None(default), invoking profile/efficiency-like filling
    :param str ax: histogram's axis name
    :param str weigthax: histogram's weight axis name
    :param concurrent.futures.ProcessPoolExecutor or concurrent.futures.ThreadPoolExecutor strategy:

    :returns: filled hist
    """

    if strategy not in (
        concurrent.futures.ProcessPoolExecutor,
        concurrent.futures.ThreadPoolExecutor,
    ):
        raise ValueError(
            "strategy can only be <concurrent.futures.ProcessPoolExecutor> or <concurrent.futures.ThreadPoolExecutor>"
        )
    histo_ = histo.cleared()

    with strategy(
        max_workers=min(len(filelist), int(multiprocessing.cpu_count() * 0.8))
    ) as executor:
        futures = {
            executor.submit(
                SingleFill,
                histo_,
                loaderCls,
                dataKey,
                fillingMethod,
                f,
                weight=weight,
                benchKey=benchKey,
                ax=ax,
                weightax=weightax,
            ): f
            for f in filelist
        }
        histoResults = []
        for future in concurrent.futures.as_completed(futures):
            fn = futures[future]
            try:
                histoResults.append(future.result())
            except Exception as e:
                print("-------------------------------------------------------------")
                print(
                    "Exception occurred in `{}`".format(uch.getFuncSignature(MultiFill))
                )
                print("File:", fn)
                print("Msg:", str(e))
                print("-------------------------------------------------------------")

        if histoResults:
            histo_ = reduce(lambda x, y: x + y, histoResults)

    return histo_


# ------------------------------------------------------------------------------


def MultiFill2D(
    histo,
    loaderCls,
    dataKeys,
    fillingMethods,
    filelist,
    weight=None,
    benchKey=None,
    ax="x",
    weightax="w",
    strategy=concurrent.futures.ProcessPoolExecutor,
):
    """
    fill 2D histogram with `fillingMethods` for all files in `filelist`.

    :param :py:class:`HistBook.Hist` histo: Hist class
    :param class `loaderCls`: loaderCls with all necessary data attributes stuffed
    :param list datakeys: data attribute names
    :param callable/list fillingMethods: transformation function(s)
    :param list filelist: a list of files to work on
    :param scalar/array-like weight: if not None(default), apply weight
    :param str benchKey: if not None(default), invoking profile/efficiency-like filling
    :param str ax: histogram's axis name
    :param str weigthax: histogram's weight axis name
    :param concurrent.futures.ProcessPoolExecutor or concurrent.futures.ThreadPoolExecutor strategy:

    :returns: filled hist
    """

    if strategy not in (
        concurrent.futures.ProcessPoolExecutor,
        concurrent.futures.ThreadPoolExecutor,
    ):
        raise ValueError(
            """strategy can only be <concurrent.futures.ProcessPoolExecutor> or <concurrent.futures.ThreadPoolExecutor>"""
        )
    histo_ = histo.cleared()

    with strategy(
        max_workers=min(len(filelist), int(multiprocessing.cpu_count() * 0.8))
    ) as executor:
        futures = {
            executor.submit(
                SingleFill2D,
                histo_,
                loaderCls,
                dataKeys,
                fillingMethods,
                f,
                weight=weight,
                benchKey=benchKey,
                ax=ax,
                weightax=weightax,
            ): f
            for f in filelist
        }
        histoResults = []
        for future in concurrent.futures.as_completed(futures):
            fn = futures[future]
            try:
                histoResults.append(future.result())
            except Exception as e:
                print("-------------------------------------------------------------")
                print(
                    "Exception occurred in `{}`".format(uch.getFuncSignature(MultiFill))
                )
                print("File:", fn)
                print("Msg:", str(e))
                print("-------------------------------------------------------------")

        if histoResults:
            histo_ = reduce(lambda x, y: x + y, histoResults)

    return histo_


# ------------------------------------------------------------------------------

# ticks Locator & Formatter do the job gracefully
# still need for pandas scatter plots
def _addMidPts(df):
    """
    add mid points to dataframe so plt knows where to plot.
    by saying *mid*, it means the **left** side of the interval
    """

    _mid = [x[0].left if isinstance(x[0], pd.Interval) else np.nan for x in df.index]
    #     _mid[0] = _mid[1]-(_mid[2]-_mid[1])
    #     if isinstance(df.index[-1][0], pd.Interval):
    #         _mid[-1] = _mid[-2]+(_mid[-2]-_mid[-3])
    #     else:
    #         _mid[-2] = _mid[-3]+(_mid[-3]-_mid[-4])
    df["midpts"] = _mid

    return df


# ------------------------------------------------------------------------------


def OverlayHistoBenched(histos, ax, xlabel=None, ylabel=None, title=None, **kwargs):
    """
    overlay a dict of :py:class:`HistBook.Hist`s on a :py:class:`matplotlib.pyplot` axis
    , with keys as labels.

    :param dict histos: a dictionary of :py:class:`HistBook.Hist` with labels as keys
    :param axis ax: ax to plot on
    :param str xlabel: if not None(default), set xlabel
    :param str ylabel: if not None(default), set ylabel
    :param str title: if not None(default), set title
    """

    if not histos:
        return ax

    _opt = dict(
        x="midpts",
        y="y",
        yerr="err(y)",
        logy=False,
        colors=plt.rcParams["axes.prop_cycle"],
    )
    _opt.update(kwargs)
    colors = _opt.pop("colors")

    dfs = [_addMidPts(h.pandas("y")) for h in histos.values()]

    for df, l, c in zip(dfs, histos, colors):
        _opt.update(c)
        df.plot.scatter(ax=ax, label=l, **_opt)

    # add labels for axis and title if told
    if xlabel:
        ax.set_xlabel(xlabel, x=1.0, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel, y=1.0, ha="right")
    if title:
        ax.set_title(title, x=0.0, ha="left")
    ax.legend()

    return ax


# ------------------------------------------------------------------------------


def prepHisto(histo, norm=False, drawunderflow=True, drawoverflow=True):
    """
    pass in a Hist object, return (bin_edges, vals, errs)
    """
    if norm:  # wont draw over/underflow bin
        drawunderflow = False
        drawoverflow = False

    df = histo.pandas(normalized=norm)
    underflowBin, overflowBin, nanBin = None, None, None
    normBins = []
    for i, b in enumerate(df.index):
        b = b[0]
        if isinstance(b, str):
            nanBin = i
        else:
            if np.isinf(float(b.left)):
                underflowBin = i
            elif np.isinf(float(b.right)):
                overflowBin = i
            else:
                normBins.append(b)

    # edges
    edges = [x.left for x in normBins] + [normBins[-1].right]
    vals = df["count()"].loc[normBins].values
    errs = df["err(count())"].loc[normBins].values
    if drawunderflow and underflowBin is not None:
        edges = [edges[0] - (edges[1] - edges[0])] + edges
        vals = np.hstack([df["count()"].iloc[underflowBin], vals])
        errs = np.hstack([df["err(count())"].iloc[underflowBin], errs])
    if drawoverflow and overflowBin is not None:
        edges += [edges[-1] + (edges[-1] - edges[-2])]
        vals = np.hstack([vals, df["count()"].iloc[overflowBin]])
        errs = np.hstack([errs, df["err(count())"].iloc[overflowBin]])
    edges = np.array(edges)
    return edges, vals, errs


# ------------------------------------------------------------------------------


def drawHisto(
    hist,
    ax,
    norm=False,
    underflow=False,
    overflow=True,
    logy=False,
    label="label",
    **kwargs
):
    """
    Draw a single Hist object.
    """
    edges, vals, errs = prepHisto(
        hist, norm=norm, drawunderflow=underflow, drawoverflow=overflow
    )

    binCenters = (edges[:-1] + edges[1:]) / 2
    line, = ax.step(x=edges, y=np.hstack([vals, vals[-1]]), where="post")
    errb = ax.errorbar(
        x=binCenters, y=vals, yerr=errs, linestyle="none", color=line.get_color()
    )
    ax.autoscale(axis="x", tight=True)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    ax.legend([(line, errb)], [label])
    xlabel_ = kwargs.get("xlabel", "x axis")
    ylabel_ = kwargs.get("ylabel", "y axis")
    title_ = kwargs.get("title", "Title")
    ax.set_xlabel(xlabel_, x=1.0, ha="right")
    ax.set_ylabel(ylabel_, y=1.0, ha="right")
    ax.set_title(title_, x=0.0, ha="left")
    return ax


# ------------------------------------------------------------------------------


def OverlayHisto(
    hists,
    ax,
    norm=False,
    underflow=False,
    overflow=True,
    logy=False,
    **kwargs
):
    """
    Draw multiple Hist objects.
    """
    if not hists:
        return ax
    legContainer = [[], []]
    for l, hist in hists.items():
        edges, vals, errs = prepHisto(
            hist, norm=norm, drawunderflow=underflow, drawoverflow=overflow
        )

        binCenters = (edges[:-1] + edges[1:]) / 2
        line, = ax.step(x=edges, y=np.hstack([vals, vals[-1]]), where="post")
        errb = ax.errorbar(
            x=binCenters, y=vals, yerr=errs, linestyle="none", color=line.get_color()
        )
        legContainer[0].append((line, errb))
        legContainer[1].append(l)
    ax.autoscale(axis="x", tight=True)
    ax.legend(*legContainer)
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    xlabel_ = kwargs.get("xlabel", "x axis")
    ylabel_ = kwargs.get("ylabel", "y axis")
    title_ = kwargs.get("title", "Title")
    ax.set_xlabel(xlabel_, x=1.0, ha="right")
    ax.set_ylabel(ylabel_, y=1.0, ha="right")
    ax.set_title(title_, x=0.0, ha="left")
    return ax


# ------------------------------------------------------------------------------


def OverlayHistoStacked(
    hists, ax, drawstack=True, underflow=False, overflow=True, logy=True, **kwargs
):
    """
    kwargs should include 'xlabel', 'ylabel', 'title'.
    if drawstack is True, kwargs should include 'label' as well.
    """
    from collections import OrderedDict

    histos = OrderedDict(
        sorted(hists.items(), key=lambda x: x[1].pandas()["count()"].sum())
    )
    data = {
        l: prepHisto(h, drawunderflow=underflow, drawoverflow=overflow)
        for l, h in histos.items()
    }
    sumdata = prepHisto(
        reduce(lambda x, y: x + y, histos.values()),
        drawunderflow=underflow,
        drawoverflow=overflow,
    )

    sum_stack = np.vstack([v[1] for v in data.values()])
    sum_stack = np.hstack([sum_stack, sum_stack[:, -1:]])
    sum_total = sum_stack.sum(axis=0)
    bin_edges = list(data.values())[0][0]
    unc = sumdata[2]
    unc = np.hstack([unc, unc[-1]])

    if drawstack:
        ax.stackplot(bin_edges, sum_stack, labels=data.keys(), step="post")
    else:
        label_ = kwargs.get("label", "Total")
        ax.step(x=bin_edges, y=sum_total, where="post", label=label_)
    ax.fill_between(
        x=bin_edges,
        y1=sum_total - unc,
        y2=sum_total + unc,
        step="post",
        label="Stat. Unc.",
        hatch="xxx",
        alpha=0.15,
    )
    ax.autoscale(axis="x", tight=True)
    ax.legend()
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, None)
    xlabel_ = kwargs.get("xlabel", "x axis")
    ylabel_ = kwargs.get("ylabel", "y axis")
    title_ = kwargs.get("title", "Title")
    ax.set_xlabel(xlabel_, x=1.0, ha="right")
    ax.set_ylabel(ylabel_, y=1.0, ha="right")
    ax.set_title(title_, x=0.0, ha="left")
    return ax


# ------------------------------------------------------------------------------


def OverlayHistoStacked_old(
    histos, ax, error=False, xlabel=None, ylabel=None, title=None, **kwargs
):
    """
    overlay a dict of :py:class:`HistBook.Hist`s on a :py:class:`matplotlib.pyplot` axis
    , with keys as labels.

    :param dict histos: a dictionary of :py:class:`HistBook.Hist` with labels as keys
    :param axis ax: ax to plot on
    :param bool error: if not False(default), draw error bars
    :param str xlabel: if not None(default), set xlabel
    :param str ylabel: if not None(default), set ylabel
    :param str title: if not None(default), set title

    :returns: ax
    """

    if not histos:
        return ax

    _opt = dict(
        y="count()",
        yerr="err(count())",
        width=1.0,
        rot=0,
        logy=True,
        colors=plt.rcParams["axes.prop_cycle"],
    )
    _opt.update(kwargs)
    if not error:
        _opt.pop("yerr")

    # dfs = [h.pandas() for h in histos.values()]
    dfs = {l: h.pandas() for l, h in histos.items()}
    dfs = OrderedDict(sorted(dfs.items(), key=lambda x: x[1]["count()"].sum()))

    cnts = [df["count()"].to_numpy() for df in dfs.values()]
    btms = [0] + [reduce(lambda x, y: x + y, cnts[:i]) for i in range(1, len(dfs))]
    colors = _opt.pop("colors")

    for l, btm, c in zip(dfs, btms, colors):
        _opt.update(c)
        dfs[l].plot.bar(ax=ax, label=l, bottom=btm, **_opt)

    # adjust ylim
    # maxBinVal = max([df['count()'].max() for df in dfs])
    # ax.set_ylim([ax.get_ylim()[0], maxBinVal*1.05])

    # adjust ticks
    numeric_const_pattern = (
        "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    )
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    origlabels = [t.get_text() for t in ax.xaxis.get_majorticklabels()]
    newlabels = [rx.findall(x) for x in origlabels]
    newlabels = ["{:g}".format(float(l[0])) for l in newlabels[1:-1]]

    ax.xaxis.set_major_locator(plticker.AutoLocator())
    nNewlabels = len(ax.xaxis.get_majorticklabels())
    newlabels = ["-inf"] + newlabels[:: int(nNewlabels + 2)] + ["inf"]
    ax.xaxis.set_major_formatter(plticker.FuncFormatter(lambda x, pos: newlabels[pos]))

    # add labels for axis and title if told
    if xlabel:
        ax.set_xlabel(xlabel, x=1.0, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel, y=1.0, ha="right")
    if title:
        ax.set_title(title, x=0.0, ha="left")
    ax.legend()

    return ax


# ------------------------------------------------------------------------------


def PlotHisto2D(
    histo,
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    benched=False,
    nticks=(7, 7),
    **kwargs
):
    """plot 2d histogram with sns

    Parameters
    ----------
    histo : histbook.Hist
        histogram container
    ax : matplotlib.pyplot.axis
        axis to plot on
    xlabel : str, optional
        name of xlabel, by default None
    ylabel : str, optional
        name of ylabel, by default None
    title : str, optional
        name of title, by default None
    benched : bool, optional
        if this is an efficiency like, by default False
    nticks : tuple, optional
        two-element tuple, number of ticks for x-axis and y-axis, by default (7,7)
    """

    if not histo:
        return ax

    histoAsDf = histo.pandas()["count()"] if not benched else histo.pandas("b")["b"]
    histoAsDD = defaultdict(dict)

    for x in histoAsDf.index.levels[0]:
        if isinstance(x, str):
            if "NaN" in x:
                continue
            xidx = int(x)
        elif isinstance(x, pd.Interval):
            if np.isinf(x.left) or np.isinf(x.right):
                continue
            xidx = x.left
        else:
            raise TypeError

        for y in histoAsDf.index.levels[1]:
            if isinstance(y, str):
                if "NaN" in y:
                    continue
                yidx = int(y)
            elif isinstance(y, pd.Interval):
                if np.isinf(y.left) or np.isinf(y.right):
                    continue
                yidx = y.left
            else:
                raise TypeError

            histoAsDD[xidx][yidx] = histoAsDf[x][y]

    nxticks, nyticks = nticks
    data = pd.DataFrame(histoAsDD).sort_index(ascending=False)

    nxticks = min(nxticks, len(data.columns))
    nyticks = min(nyticks, len(data.index))

    import math

    xticks_step = math.ceil(len(data.columns) / nxticks)
    yticks_step = math.ceil(len(data.index) / nyticks)
    kwargs.update(dict(xticklabels=xticks_step, yticklabels=yticks_step, ax=ax))

    import seaborn as sns

    sns.heatmap(data, **kwargs)

    if xlabel:
        ax.set_xlabel(xlabel, x=1.0, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel, y=1.0, ha="right")
    if title:
        ax.set_title(title, x=0.0, ha="left")

    return ax


# ------------------------------------------------------------------------------


def OverlayHisto_old(
    histos, ax, norm=True, xlabel=None, ylabel=None, title=None, **kwargs
):
    """
    overlay a dict of :py:class:`HistBook.Hist`s on a :py:class:`matplotlib.pyplot` axis
    , with keys as labels.

    :param dict histos: a dictionary of :py:class:`HistBook.Hist` with labels as keys
    :param axis ax: ax to plot on
    :param bool norm: if True(default), normalize histograms
    :param str xlabel: if not None(default), set xlabel
    :param str ylabel: if not None(default), set ylabel
    :param str title: if not None(default), set title

    :returns: ax
    """

    if not histos:
        return ax

    _opt = dict(y="count()", yerr="err(count())", drawstyle="steps-mid", logy=True)
    _opt.update(kwargs)

    for l, h in histos.items():
        # h.pandas(normalized=norm).plot.line(ax=ax, label=l, **_opt)
        # we drop the {NaN} bin 'cause we do not plot it
        df_ = h.pandas(normalized=norm)
        if isinstance(df_.index[-1][0], str) and "NaN" in df_.index[-1][0]:  # ('{NaN}')
            df_ = df_.iloc[:-1]

        edges_ = [x[0] for x in df_.index] + [df_.index[-1][0]]
        edges_ = list(
            map(lambda x: x.left if isinstance(x, pd.Interval) else float(x), edges_)
        )
        # edges_ = [x[0].left for x in df_.index if isinstance(x[0], pd.Interval)]
        # edges_ += [df_.index[-1][0].left]
        yval_ = np.hstack([df_[_opt["y"]], df_[_opt["y"]][-1]])
        yerr_ = np.hstack([df_[_opt["yerr"]], df_[_opt["yerr"]][-1]])
        ax.errorbar(x=edges_, y=yval_, yerr=yerr_, drawstyle=_opt["drawstyle"], label=l)

    # adjust ticks
    # numeric_const_pattern = (
    #     "[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?"
    # )
    # rx = re.compile(numeric_const_pattern, re.VERBOSE)

    # ax.xaxis.set_major_locator(plticker.AutoLocator())
    # origlabels = [t.get_text() for t in ax.xaxis.get_majorticklabels()]
    # newlabels = []
    # for ol in origlabels:
    #     numfound = rx.findall(ol)
    #     if "-inf" in ol:
    #         newlabels.append("uf {:g}".format(float(numfound[-1])))
    #     elif "inf" in ol:
    #         newlabels.append("{:g} of".format(float(numfound[-1])))
    #     else:
    #         newlabels.append("{:g}".format(float(numfound[-1])) if numfound else "")
    # ax.xaxis.set_major_formatter(plticker.FuncFormatter(lambda x, pos: newlabels[pos]))

    # add labels for axis and title if told
    if xlabel:
        ax.set_xlabel(xlabel, x=1.0, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel, y=1.0, ha="right")
    if title:
        ax.set_title(title, x=0.0, ha="left")
    if _opt["logy"]:
        ax.set_yscale("log")
    ax.autoscale(axis="x", tight=True)
    ax.legend()

    return ax


# ------------------------------------------------------------------------------


class ffLoader(MutableMapping):
    def __init__(
        self,
        filename,
        *args,
        trigpaths=[
            "HLT_DoubleL2Mu23NoVtx_2Cha",
            "HLT_DoubleL2Mu23NoVtx_2Cha_NoL2Matched",
            "HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed",
            "HLT_DoubleL2Mu23NoVtx_2Cha_CosmicSeed_NoL2Matched",
            "HLT_DoubleL2Mu25NoVtx_2Cha_Eta2p4",
            "HLT_DoubleL2Mu25NoVtx_2Cha_CosmicSeed_Eta2p4",
        ],
        **kwargs
    ):
        self._storage = dict(*args, **kwargs)
        self._storage["filename"] = filename
        self._storage["tree"] = uuh.GetFfNtupleTree(filename)
        self._storage["trigpaths"] = trigpaths
        self._storage["mHLT"] = np.logical_or.reduce(
            list(map(lambda p: self._storage["tree"][p].array(), trigpaths))
        )

    def __getitem__(self, key):
        return self._storage[key]

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __delitem__(self, key):
        self._storage.pop(key)

    def __iter__(self):
        return iter(self._storage)

    def __len__(self):
        return len(self._storage)

    @property
    def tree(self):
        return self._storage["tree"]

    @property
    def mHLT(self):
        return self._storage["mHLT"]


# ------------------------------------------------------------------------------


class ffFiller:
    def __init__(self, fillingMethod, aloader):
        self._fillingMethod = fillingMethod
        self._loader = aloader

    def fill(self, histo, dataKey, ax="x", weight=None, weightax="w"):
        try:
            dataToFill = {ax: self._fillingMethod(self._loader[dataKey]())}
            if weight is not None:
                dataToFill.update({weightax: weight})
            else:
                if weightax in histo.fields:
                    dataToFill.update({weightax: 1})
            histo.fill(dataToFill)
        except KeyError as e:
            print("ERROR: {} is not a valid branch name.".format(dataKey))
            print("Exception msg: ", str(e))
            raise

    def fillp(self, histo, dataKey, benchKey):
        try:
            dataToFill = dict(
                x=self._fillingMethod(self._loader[dataKey]()),
                y=self._fillingMethod(self._loader[benchKey]()),
            )
            histo.fill(dataToFill)
        except KeyError as e:
            print(
                "ERROR: ({}, {}) is not a valid branch name.".format(dataKey, benchKey)
            )
            print("Exception msg: ", str(e))
            raise


# ------------------------------------------------------------------------------


class ffFiller2D:
    def __init__(self, fillingMethods, aLoader):
        self._fillingMethods = fillingMethods
        self._loader = aLoader

    def fill(self, histo, dataKeys, ax=["x", "y"], weight=None, weightax="w"):
        assert hasattr(ax, "__len__") and len(ax) == 2
        try:
            dataToFill = dict()
            fillMethods = []
            if callable(self._fillingMethods):
                fillMethods = [self._fillingMethods] * len(dataKeys)
            elif hasattr(self._fillingMethods, "__len__"):
                assert len(self._fillingMethods) == len(dataKeys)
                fillMethods = self._fillingMethods
            else:
                print(
                    "<ffFiller2D._fillingMethods> can only be either callable or list-type of callable."
                )
                raise TypeError

            dataToFill = dict(
                zip(
                    ax,
                    [_fm(self._loader[k]()) for _fm, k in zip(fillMethods, dataKeys)],
                )
            )

            if weight is not None:
                dataToFill.update({weightax: weight})
            else:
                if weightax in histo.fields:
                    dataToFill.update({weightax: 1})

            histo.fill(dataToFill)
        except KeyError as e:
            print("ERROR: {} are not valid branch names.".format(dataKeys))
            print("Exception msg: ", str(e))
            raise

    def fillp(self, histo, dataKeys, benchKey, ax=["x", "y"]):
        try:
            dataToFill = dict()
            fillMethods = []
            if callable(self._fillingMethods):
                fillMethods = [self._fillingMethods] * (len(dataKeys) + 1)
            elif hasattr(self._fillingMethods, "__len__"):
                assert len(self._fillingMethods) == len(dataKeys) + 1
                fillMethods = self._fillingMethods
            else:
                print(
                    "<ffFiller2D._fillingMethods> can only be either callable or list-type of callable."
                )
                raise TypeError

            dataToFill = dict(
                zip(
                    ax + ["b"],
                    [
                        _fm(self._loader[k]())
                        for _fm, k in zip(fillMethods, dataKeys + [benchKey])
                    ],
                )
            )

            histo.fill(dataToFill)
        except KeyError as e:
            print("ERROR: {} are not valid branch names.".format(dataKeys))
            print("Exception msg: ", str(e))
            raise


# ------------------------------------------------------------------------------


def dressingSubDetBoundary(ax):
    abspos_ = [2.9, 16, 112, 280, 740]
    tags_ = ["Beampipe", "Pixel", "Tracker", "E/HCAL", "MS"]
    xmin, xmax = ax.get_xlim()
    postag = [x for x in zip(abspos_, tags_) if x[0] >= xmin and x[0] <= xmax]
    ax.vlines(
        (np.array([x[0] for x in postag]) - xmin) / (xmax - xmin),
        0,
        1,
        linestyles="dashdot",
        color="grey",
        transform=ax.transAxes,
        linewidth=1,
    )
    for xpos, tag in postag:
        ax.text(
            (xpos - xmin) / (xmax - xmin) - 0.01,
            0.01,
            tag,
            ha="right",
            va="bottom",
            rotation=90,
            transform=ax.transAxes,
            color="grey",
        )
    return ax
