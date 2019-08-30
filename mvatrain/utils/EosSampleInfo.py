#!/usr/bin/env python
import os
import sys
import subprocess
from datetime import datetime
from collections import OrderedDict
import utils.commonHelpers as uch

tsfmt = "%y%m%d_%H%M%S"


class EosSampleInfo:
    def __init__(self, path):
        _statusCheck = subprocess.getstatusoutput(f"eos ls -s {path}")
        if _statusCheck[0] != 0:
            sys.exit(f"{path} does NOT exist in EOS, exit.")

        self.fPath = path
        self.fUsage = subprocess.getoutput(f"eosdu -h {path}")
        if "command not found" in self.fUsage:
            print("Need to use `eosdu` utility to get disk usage.")
            sys.exit("source /cvmfs/cms-lpc.opensciencegrid.org/FNALLPC/lpc-scripts")
        _subdirs = uch.eosls(path)
        self.fValid = True
        try:
            _subdirs = sorted([datetime.strptime(d, tsfmt) for d in _subdirs])
            self.fVersions = [d.strftime(tsfmt) for d in _subdirs]
        except ValueError:
            self.fValid = False
            self.fVersions = []
        self.fVersionUsage = map(
            lambda x: subprocess.getoutput(
                "eosdu -h {}".format(os.path.join(self.fPath, x))
            ),
            self.fVersions,
        )

    @property
    def name(self):
        return self.fPath

    @property
    def usage(self):
        return self.fUsage

    @property
    def isValid(self):
        return self.fValid

    @property
    def _subdirs(self):
        return uch.eosls(self.fPath)

    @property
    def versions(self):
        return self.fVersions

    @property
    def versionUsage(self):
        return OrderedDict(zip(self.fVersions, self.fVersionUsage))

    @property
    def absSubDirs(self):
        return [os.path.join(self.fPath, d) for d in self.fVersions]

    def getVersion(self, i, full=False):
        try:
            res = self.fVersions[i]
            if full:
                res = os.path.join(self.fPath, res)
            return res
        except IndexError:
            print(f"ERROR: verison {i} NOT exist!")
            print(f"Availables are: {self.fVersions}")
            return ""

    def firstOnes(self, i, action):
        assert i >= 0
        if action == "keep":
            return self.absSubDirs[:i]
        elif action == "drop":
            return self.absSubDirs[i:]
        else:
            print(f"Unknown action - {action}")
            return []

    def lastOnes(self, i, action):
        i = -i
        assert i < 0
        if action == "keep":
            return self.absSubDirs[i:]
        elif action == "drop":
            return self.absSubDirs[:i]
        else:
            print(f"Unknown action - {action}")
            return []


def main():
    testSamplePath = "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/CRAB_PrivateMC/SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-0p25_ctau-1p875"
    sinfo = EosSampleInfo(testSamplePath)

    print("[name]:\t", sinfo.name)
    print("[usage]:\t", sinfo.usage)
    print("[versions]:", *list(sinfo.versionUsage.items()), sep="\n")
    print("[subdirs]:", *sinfo.absSubDirs, sep="\n")
    print("[First2Vers] - keep:", *sinfo.firstOnes(2, "keep"), sep="\n")
    print("[First2Vers] - drop:", *sinfo.firstOnes(2, "drop"), sep="\n")
    print("[Last2Vers] - keep:", *sinfo.lastOnes(2, "keep"), sep="\n")
    print("[Last2Vers] - drop:", *sinfo.lastOnes(2, "drop"), sep="\n")


if __name__ == "__main__":
    main()
