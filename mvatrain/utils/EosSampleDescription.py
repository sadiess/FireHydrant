#!/usr/bin/env python
import os
import sys
import subprocess
import utils.commonHelpers as uch
from utils.EosSampleInfo import EosSampleInfo


class EosSampleDescription:
    def __init__(self, path):
        _statusCheck = subprocess.getstatusoutput(f"eos ls -s {path}")
        if _statusCheck[0] != 0:
            sys.exit(f"{path} does NOT exist in EOS, exit.")

        self.fPath = path
        _sampleList = [os.path.join(path, d) for d in uch.eosls(path)]
        _sampleList = [EosSampleInfo(s) for s in _sampleList]
        self.fSampleList = [s for s in _sampleList if s.isValid]
        self.fUnhealthySamples = [s for s in _sampleList if not s.isValid]

    @property
    def name(self):
        return self.fPath

    @property
    def samples(self):
        return [s.name for s in self.fSampleList]

    @property
    def unhealthySamples(self):
        return [s.name for s in self.fUnhealthySamples]

    def printSamples(self, stdout=True):
        lnfmt = "{:>12} {:3} vers {}"
        res = [lnfmt.format(s.usage, len(s.versions), s.name) for s in self.fSampleList]
        if stdout:
            print(*res, sep="\n")
            return
        else:
            return "\n".join(res)

    def printUnhealthySamples(self, stdout=True):
        lnfmt = "{}\n\t{}"
        res = [lnfmt.format(s.name, s._subdirs) for s in self.fUnhealthySamples]
        if stdout:
            print(*res, sep="\n")
            return
        else:
            return "\n".join(res)

    def getVersion(self, i):
        return [d.getVersion(i, full=True) for d in self.fSampleList]


def main():
    SAMPLE_ROOT = "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/"
    sigpath = SAMPLE_ROOT + "CRAB_PrivateMC/"
    bkgpath = SAMPLE_ROOT + "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8"
    bkgpath2 = SAMPLE_ROOT + "JpsiToMuMu_JpsiPt8_TuneCP5_13TeV-pythia8"

    sigSampleDesc = EosSampleDescription(sigpath)
    bkgSampleDesc = EosSampleDescription(bkgpath)
    bkgSampleDesc2 = EosSampleDescription(bkgpath2)

    sigSampleDesc.printSamples()
    print("-" * 79)
    sigSampleDesc.printUnhealthySamples()
    print("="*79)

    bkgSampleDesc.printSamples()
    print("-" * 79)
    bkgSampleDesc.printUnhealthySamples()
    print("="*79)

    bkgSampleDesc2.printSamples()
    print("-" * 79)
    bkgSampleDesc2.printUnhealthySamples()
    print("="*79)

if __name__ == "__main__":
    main()
