#!/usr/bin/env python
import os
import sys
import getpass
import subprocess
from datetime import datetime
from functools import partial
from collections import namedtuple
from inspect import signature
import concurrent.futures

import numpy as np

sample = namedtuple("sample", "mxx mdp ctau proc")


def getFuncSignature(func):
    return func.__name__ + str(signature(func))


def dropna(arr):
    """Given a numpy array, return it after filtering out
       nan values
    """

    return arr[~np.isnan(arr)]


def eosls(eospath, xdirector="root://cmseos.fnal.gov/"):

    if eospath.startswith("root://"):
        cmd = "eos ls {}".format(eospath)
    else:
        cmd = "eos {0} ls {1}".format(xdirector, eospath)

    try:
        return subprocess.check_output(cmd.split()).decode().split()
    except:
        print(f"ERROR when calling: {cmd}")
        return []


def eosfindfile(eospath, xdirector="root://cmseos.fnal.gov/"):

    if eospath.startswith("root://"):
        cmd = "eos find -f --xurl {}".format(eospath)
    else:
        cmd = "eos {0} find -f --xurl {1}".format(xdirector, eospath)

    return subprocess.check_output(cmd.split()).decode().split()


def eoscp(src, dest):

    if not dest.startswith("root://"):
        try:
            os.makedirs(os.path.dirname(dest))
        except:
            pass

    cmd = "eos cp -s {0} {1}".format(src, dest)
    signal = subprocess.run(cmd.split())
    try:
        signal.check_returncode()
    except Exception as e:
        print("System call '{}' failed!".format(cmd), str(e))
        print("return code: ", signal.returncode)
        print("stdout: ", signal.returncode)
        print("stderr: ", signal.stderr)
        raise

    return signal.returncode


def eosmkdir(dir):

    cmd = "eos mkdir {}".format(dir)
    signal = subprocess.run(cmd.split())
    try:
        signal.check_returncode()
    except Exception as e:
        print("System call '{}' failed!".format(cmd), str(e))
        print("return code: ", signal.returncode)
        print("stdout: ", signal.returncode)
        print("stderr: ", signal.stderr)
        raise

    return signal.returncode


def eosmv(src, dest):

    cmd = "eos mv -s {} {}".format(src, dest)
    signal = subprocess.run(cmd.split())
    try:
        signal.check_returncode()
    except Exception as e:
        print("System call '{}' failed!".format(cmd), str(e))
        print("return code: ", signal.returncode)
        print("stdout: ", signal.returncode)
        print("stderr: ", signal.stderr)
        raise

    return signal.returncode


def eosrm(fOrd, xdirector="root://cmseos.fnal.gov/"):

    if fOrd.startswith("root://"):
        cmd = "eos rm -r {}".format(fOrd)
    else:
        cmd = "eos {} rm -r {}".format(xdirector, fOrd)

    signal = subprocess.run(cmd.split())
    try:
        signal.check_returncode()
    except Exception as e:
        print("System call '{}' failed!".format(cmd), str(e))
        print("return code: ", signal.returncode)
        print("stdout: ", signal.returncode)
        print("stderr: ", signal.stderr)
        raise

    return signal.returncode


def xrdcpdir(src, dest, xdirector="root://cmseos.fnal.gov/", force=True):

    if not src.startswith("root://"):
        src = xdirector + src
    if not dest.startswith("root://"):
        dest = xdirector + dest

    if force:
        cmd = "xrdcp -rfs --parallel 4 {} {}".format(src, dest)
    else:
        cmd = "xrdcp -rs --parallel 4 {} {}".format(src, dest)

    signal = subprocess.run(cmd.split())
    try:
        signal.check_returncode()
    except Exception as e:
        print("System call '{}' failed!".format(cmd), str(e))
        print("return code: ", signal.returncode)
        print("stdout: ", signal.returncode)
        print("stderr: ", signal.stderr)
        raise

    return signal.returncode


def mergeSingleCrabOutput(
    path, nth=-1, xdirector="root://cmseos.fnal.gov/", force=True
):
    """
    Merge output of crab jobs into a single ROOT file.

    :param str path: xrootd LFN path, its child directories should be
    timestamp strings
    :param int nth: select n-th batch to merge. Default -1 refers to the latest
    :param str xdirector: xrootd director
    :param bool force: force to hadd
    :return: merged file LFN path

    :rtype: str
    """

    timestamp_fmt = "%y%m%d_%H%M%S"
    try:
        sortedTimestampDirs = sorted(
            [
                datetime.strptime(d, timestamp_fmt)
                for d in eosls(path, xdirector=xdirector)
            ]
        )
        timestamp = sortedTimestampDirs[nth].strftime(timestamp_fmt)
        timestampedDir = os.path.join(path, timestamp)

        res = os.path.join(timestampedDir, os.path.basename(path) + ".root")
        if not res.startswith("root://"):
            res = xdirector + res

        tmpdir = "/uscmst1b_scratch/lpc1/3DayLifetime/{}".format(getpass.getuser())
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        tmpres = os.path.join(tmpdir, timestamp + os.path.basename(res))
        if os.path.exists(tmpres):
            os.remove(tmpres)

        src = [
            f
            for f in eosfindfile(timestampedDir, xdirector=xdirector)
            if f.endswith(".root") and f.split("/")[-2].isnumeric()
        ]
        oldMergedFile = [
            os.path.join(timestampedDir, f)
            for f in eosls(timestampedDir)
            if f.endswith(".root")
        ]

        DEVNULL = open(os.devnull, "wb")

        if force:
            if len(oldMergedFile):
                print("already hadded! removing..", ", ".join(oldMergedFile))
                for f in oldMergedFile:
                    eosrm(f)
        else:
            if res in oldMergedFile:
                print(
                    "Target file - {} already exsits! Toggle `force=True` if you want to re-hadd. Passing..".format(
                        res
                    )
                )
                return res

        cmd = "hadd -f {0} {1}".format(tmpres, " ".join(src))
        subprocess.run(cmd.split(), stdout=DEVNULL, stderr=subprocess.STDOUT)
        eoscp(tmpres, res)
        os.remove(tmpres)
        return res

    except ValueError:
        sys.exit(path + " contains non-timestamp-like child directory!")
    except IndexError:
        sys.exit(path + " does not have the asked " + str(nth) + " -th directory.")


def mergeMultipleCrabOutputs(
    paths, nth=-1, xdirector="root://cmseos.fnal.gov/", force=False
):
    """
    Merge multiple outputs of crab jobs parallelly.

    :param list paths: xrootd LFN paths, its grandchild directories should be
    timestamp strings
    :param int nth: select n-th batch to merge. Default -1 refers to the latest
    :param str xdirector: xrootd director :param bool
    force: force to re-hadd :return: list of merged file LFN paths

    :rtype: list
    """

    worker = partial(mergeSingleCrabOutput, nth=nth, xdirector=xdirector, force=force)

    res = list()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(paths)) as executor:
        for _, file in zip(paths, executor.map(worker, paths)):
            res.append(file)

    return res


def mergeGroupCrabOutputs(
    path, nth=-1, xdirector="root://cmseos.fnal.gov/", force=False
):
    """
    Merge multiple outputs of a group of crab jobs (have same pd) parallelly.

    :param str path: xrootd LFN path, its grandchild directories should be
    timestamp strings (ends with pd)
    :param int nth: select n-th batch to merge. Default -1 refers to the latest.
    :param str xdirector: xrootd director
    :param bool force: force to re-hadd
    :return: list of merged file LFN paths

    :rtype: list
    """

    keys = eosls(path, xdirector=xdirector)
    singlepaths = [os.path.join(path, d) for d in keys]

    res = mergeMultipleCrabOutputs(
        singlepaths, nth=nth, xdirector=xdirector, force=force
    )

    return res


def getNTupleFiles(
    gen_batch=-1,
    year=[2017, 2018],
    prefix="/store/user/wsi/MCSIDM/ffNtuple",
    xrootd=True,
):

    subdirs = []
    for y in year:
        year_dir = os.path.join(prefix, str(y), "CRAB_PrivateMC")
        data_dir = eosls(year_dir)
        subdirs.extend([os.path.join(year_dir, x) for x in data_dir])

    res = []

    def fetchIndiv(res, gen_batch, sd):
        sub_sd = eosls(sd)
        tsdir = sorted([datetime.strptime(x, "%y%m%d_%H%M%S") for x in sub_sd])
        try:
            res.append(
                os.path.join(
                    sd, tsdir[gen_batch].strftime("%y%m%d_%H%M%S"), "ffNtuple.root"
                )
            )
        except IndexError:
            sys.exit(
                "Directory {0} has only {1} subdirectories, \
            the requested {2}-th does NOT exist.\nExiting...".format(
                    tsdir, len(tsdir), gen_batch
                )
            )

    from functools import partial

    fetchIndividual = partial(fetchIndiv, res, gen_batch)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(subdirs)) as executor:
        executor.map(fetchIndividual, subdirs)

    if xrootd:
        res = ["root://cmseos.fnal.gov/" + f for f in res]
    return res


def getSampleParameterFromNametag(nt):
    """
    :param str nt: nametag, e.g. SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-1p2_ctau-48
    :rtype namedtuple namedtuple('sample', 'mxx mdp ctau')
    """

    pieces = nt.split("_")
    _mxx = float([x for x in pieces if "MBs" in x][0].split("-")[-1])
    _mdp = float([x for x in pieces if "MDp" in x][0].split("-")[-1].replace("p", "."))
    _ctau = float(
        [x for x in pieces if "ctau" in x][0].split("-")[-1].replace("p", ".")
    )

    return sample(mxx=_mxx, mdp=_mdp, ctau=_ctau, proc=pieces[1])


def locateNth(timestamps, nth=-1, fmt="%y%m%d_%H%M%S"):
    """
    :param list(str) timestamps: a list of string stamps
    :param int nth: nth that to be located, default=-1 latest
    :param str fmt: timestamp format
    :rtypes str: located elemetns in timestamps
    """

    from datetime import datetime

    sortedTimestamps = sorted([datetime.strptime(x, fmt) for x in timestamps])
    try:
        return sortedTimestamps[nth].strftime(fmt)
    except IndexError:
        print(
            "There are only {} timestamp folders, the requested {}th does not exist!".format(
                len(timestamps), nth
            )
        )
        raise


def getTimestampedDir(sampleDir, nth=-1):
    return os.path.join(sampleDir, locateNth(eosls(sampleDir), nth=nth))


def transferFromEosToLocal(lfn, force=False):
    """
    transfer file from EOS to 3dayNoBackupArea
    """

    dest = "/uscmst1b_scratch/lpc1/3DayLifetime/wsi"
    if not os.path.exists(dest):
        os.makedirs(dest)

    fn = os.path.basename(lfn)
    res = os.path.join(dest, fn)
    if fn in os.listdir(dest) and force:
        print("{} already exists And no override, passing..".format(fn))
        return res

    eoscp(lfn, res)
    return res


def getffNtuples(jobpath, validOnly=True):
    res = [f for f in eosfindfile(jobpath) if f.endswith(".root")]
    if validOnly:
        res = list(filter(lambda f: f.rsplit('/', 2)[-2] != 'failed', res))
    return res
