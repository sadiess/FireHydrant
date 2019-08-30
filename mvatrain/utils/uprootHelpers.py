#!/usr/bin/env python
from functools import partial

import uproot
import numpy as np
from awkward import JaggedArray
from uproot_methods import TLorentzVectorArray
from uproot_methods import TVector3Array


def NestNestObjArrayToJagged(objarr):
    """uproot read vector<vector<number>> TBranch
       as objectArray, this function convert it
       to JaggedJaggedArray
    """

    # jaggedArray of lists
    jaggedList = JaggedArray.fromiter(objarr)
    # flat to 1 level
    _jagged = JaggedArray.fromiter(jaggedList.content)

    return JaggedArray.fromoffsets(jaggedList.offsets, _jagged)


def AverageOutNestNestArray(nna):
    """
    average 2-level JaggeddArray of numbers to 1-level

    :param JaggedArray nna: input 2-level JaggedArray
    :returns: 1-level JaggedArray
    """

    nna_c = nna.deepcopy()
    nna_c.content.content = np.nan_to_num(nna_c.content.content)
    return nna_c.sum() / nna_c.count_nonzero()


''' just a&b&c
def _jaggedArray_and(jaA, jaB):
    """logic and of 2 jagged array
    """

    assert(any(jaA.starts == jaB.starts))
    assert(jaA.shape == jaB.shape)

    content = jaA.content & jaB.content
    offsets = np.append(jaA.starts, [jaA.stops[-1]])
    return JaggedArray.fromoffsets(offsets, content)


def jaggedArray_and(jlist):
    """logic and of a list of jagged array
    """

    assert(len(jlist)>1)
    if len(jlist)==2:
        return _jaggedArray_and(jlist[0], jlist[1])
    return _jaggedArray_and(jlist[0], jaggedArray_and(jlist[1:]))
'''


def p4Array(inputs):
    """
    Construct a `TLorentzVectorArray` from  given elements.

    :param inputs: :py:class:`uproot.rootio.TBranchElement` or tuple(dict, str)
    :returns: `TLorentzVectorArray`
    """

    if isinstance(inputs, uproot.tree.TBranchMethods):
        return TLorentzVectorArray.from_cartesian(*[b.array() for b in inputs.values()])
    else:
        try:
            iter(inputs)
            d, b = inputs
            assert isinstance(d, dict)
            assert isinstance(b, str)

            components = ["{}.fCoordinates.f{}".format(b, v) for v in list("XYZT")]
            if not all([k in d.keys() for k in components]):
                raise ValueError(
                    "`p4Array` fails: required keys {} are not found in {} keys.".format(
                        components, repr(d)
                    )
                )

            return TLorentzVectorArray.from_cartesian(*[d[c] for c in components])
        except TypeError as e:
            print("`p4Array()` fails with inputs: {}\n".format(str(inputs)))
            raise


def p3Array(inputs):
    """
    Construct a `TVector3Array` from given elements.

    :param inputs: :py:class:`uproot.rootio.TBranchElement` or tuple(dict, str)
    :returns: `TVector3Array`
    """

    if isinstance(inputs, uproot.tree.TBranchMethods):
        return TVector3Array.from_cartesian(*[b.array() for b in inputs.values()])
    else:
        try:
            iter(inputs)
            d, b = inputs
            assert isinstance(d, dict)
            assert isinstance(b, str)

            components = ["{}.fCoordinates.f{}".format(b, v) for v in list("XYZ")]
            if not all([k in d.keys() for k in components]):
                raise ValueError(
                    "`p3Array` fails: required keys {} are not found in {} keys.".format(
                        components, repr(d)
                    )
                )

            return TVector3Array.from_cartesian(*[d[c] for c in components])
        except TypeError as e:
            print("`p3Array()` fails with inputs: {}\n".format(str(inputs)))
            raise


def DeltaRFromMatching(srcArr, matchArr):
    """
    cross calculate dR of two Jagged TLorentzVectorArray, return the min value.

    srcArr  : JaggedArray(TLorentzVector)
    matchArr: JaggedArray(TLorentzVector)
    returns : JaggedArray
    """

    pairs = JaggedArray.cross(srcArr, matchArr, nested=True)
    metric = pairs.i0.delta_r(pairs.i1)

    return metric.min()


def MatchArrayByDeltaR(srcArr, matchArr, maxdR=0.3):
    """
    srcArr:   JaggedArray(TLorentzVector)
    matchArr: JaggedArray(TLorentzVector)
    returns:  JaggedArray (of pairs of TLorentzVector(src, match))
    """

    pairs = JaggedArray.cross(srcArr, matchArr, nested=True)
    metric = pairs.i0.delta_r(pairs.i1)
    index_of_minimized = metric.argmin()
    passes_cut = metric[index_of_minimized] < maxdR
    best_pairs_that_pass_cut = pairs[index_of_minimized][passes_cut]

    return best_pairs_that_pass_cut.flatten(axis=1)


def CrossMaskFromMatching(srcArr, matchArr, maxdR=0.3):
    """
    srcArr:   JaggedArray(TLorentzVector)
    matchArr: JaggedArray(TLorentzVector)
    returns:  JaggedArray (to be applied on
    JaggedArray.cross([srcArr], [matchArr], nested=True),
    '[]' here means same shape)
    """

    pairs = JaggedArray.cross(srcArr, matchArr, nested=True)
    metric = pairs.i0.delta_r(pairs.i1)
    index_of_minimized = metric.argmin()
    passes_cut = metric[index_of_minimized] < maxdR
    index_that_pass_cut = index_of_minimized[passes_cut]

    return index_that_pass_cut


def IndexArraysFromMatching(srcArr, matchArr, maxdR=0.3):
    """
    Matching `srcArr` elements with `matchArr` elements by deltaR, accept pairs
    only with maximum `maxdR`, returns two index arrays which to be applied on
    `srcArr` or :py:class:`JaggedArray` with same structure and `matchArr` or
    :py:class:`JaggedArray` with same structure, this should result the best
    matched pair.

    :param JaggedArray(TLorentzVector) srcArr:
    :param JaggedArray(TLorentzVector) matchArr:
    :param float maxdR:
    :returns: a pair of jagged index arrays
    """

    assert isinstance(srcArr, JaggedArray)
    assert isinstance(matchArr, JaggedArray)

    index_of_minimized = CrossMaskFromMatching(srcArr, matchArr, maxdR=maxdR)

    indexMask = JaggedArray.fromcounts(
        index_of_minimized.counts, index_of_minimized.content.counts
    ).astype(bool)
    srcIdxArr = index_of_minimized.index[indexMask]
    matIdxArr = index_of_minimized.flatten(axis=1)

    return (srcIdxArr, matIdxArr)


def MaskArrayFromIndexArray(idxArr, srcArr):
    """
    Generate a mask array from the source array(srcArr), and the corresponding
    index array(idxArr). Reverse to get the unindexed mask array.

    Parameters
    ----------
    idxArr : JaggedArray
        index array
    srcArr : JaggedArray
        source array

    """

    srcContent = np.array([False] * len(srcArr.content))
    contentIdx = (RepeatArray(srcArr.starts, idxArr) + idxArr).flatten()
    srcContent[contentIdx] = True

    res = JaggedArray.fromoffsets(srcArr.offsets, srcContent)

    return res


def MaskArraysFromMatching(srcArr, matchArr, maxdR=0.3):
    """
    srcArr:   JaggedArray(TLorentzVector)
    matchArr: JaggedArray(TLorentzVector)
    returns: (JaggedArray (that should select the matched ones from srcArr),
              JaggedArray (that should select the matched ones from matchArr))
    """

    srcIdxArr, matIdxArr = IndexArraysFromMatching(srcArr, matchArr, maxdR=maxdR)

    srcMask = MaskArrayFromIndexArray(srcIdxArr, srcArr)
    matMask = MaskArrayFromIndexArray(matIdxArr, matchArr)

    return (srcMask, matMask)


def PairArraysWithCrossMask(srcArr, matchArr, mSrcXMat):
    """
    :param JaggedArray srcArr: first jaggedArray
    :param JaggedArray matchArr: second jaggedArray
    :param JaggedArray mSrcXMat: index array that will select pairs from
        the cartisian product of JaggedArray.cross(srcArr, matchArr)

        The srcArr and matchArr does not need to result mSrcXMat,
        but the result of JaggedArray.cross(srcArr, matchArr) must have the
        same structure as mSrcXMat

    :returns: JaggedArray of pairs (src, mat)
    """

    assert isinstance(srcArr, JaggedArray)
    assert isinstance(matchArr, JaggedArray)
    assert isinstance(mSrcXMat, JaggedArray)

    return JaggedArray.cross(srcArr, matchArr, nested=True)[mSrcXMat].flatten(axis=1)


def RepeatArray(srcArr, strucArr):
    """
    replicate `srcArr` to the same length of `strucArr`, perserve structure.

    :param array-like srcArr: array to be expanded, dimension N.
    :param JaggedArray strucArr: template array whose structure to be cloned,
    dimension (N+1)

    :returns: :py:class`JaggedArray`
    """

    assert isinstance(strucArr, JaggedArray)
    assert len(srcArr) == len(strucArr)

    # res = strucArr.copy()
    # res.content = ((res.content == np.nan) | (res.content != np.nan)).astype(int)
    res = JaggedArray.fromoffsets(strucArr.offsets, [1] * len(strucArr.content))
    res = srcArr * res

    return res


def ArgNth(ja, nth=1):
    """
    nth maximum value's index of JaggedArray. Note, not nested JaggedArray yet.

    :param JaggedArray ja: input jaggedArray.
    :param int nth: n-th maximum value's index
    :return JaggedArray:
    """

    _has = (ja.counts >= nth).astype(int)
    _content = []

    for b, e in zip(ja.starts, ja.stops):
        if e - b < nth:
            continue
        subarr = ja.content[b:e]
        sortidx = np.argsort(subarr)
        _content.append(np.where(sortidx == (len(sortidx) - nth))[0][0])

    _stops = _has.cumsum()
    _starts = _stops - _has

    return JaggedArray(_starts, _stops, _content)


ArgSubMax = partial(ArgNth, nth=2)


def GetFfNtupleTree(fpath):
    """
    return an uproot.tree from the file path.

    :param str fpath: file path
    :return uproot.tree:
    """

    f_ = uproot.open(fpath)
    key_ = f_.allkeys(filtername=lambda x: x.endswith(b"ffNtuple"))[0]
    return f_[key_]
