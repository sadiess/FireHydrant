#!/usr/bin/env python
import os
import numbers
import utils.commonHelpers as uch

FFNTUPLE_ROOTDIR = "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/CRAB_PrivateMC/"


def getSampleDirInEOS(proc, mXXs, mAs, rootdir=FFNTUPLE_ROOTDIR):
    res = {}
    pool = uch.eosls(rootdir)
    for xx in mXXs:
        for A in mAs:
            key = "mXX-{}_mA-{}".format(xx, A)
            sampleNametag = [
                x for x in pool if proc in x and "MBs-{}_MDp-{}".format(xx, A) in x
            ][0]
            res[key] = os.path.join(rootdir, sampleNametag)
    return res


class SignalDescription:
    def __init__(self):
        self.fFinalStates = ["4mu", "2mu2e"]
        self.fmXX = [100, 150, 200, 500, 800, 1000]
        self.fmA = [0.25, 1.2, 5]

    @property
    def finalstates(self):
        return self.fFinalStates

    @property
    def mXX(self):
        return self.fmXX

    @property
    def mA(self):
        return self.fmA

    def files(self, fs, mXX=None, mA=None, nth=-1, grouped=False):
        res = []
        fs = fs.lower()
        if fs not in self.fFinalStates:
            print('Final state "{}" not available!'.format(fs))
            print("Choose from {}".format(self.fFinalStates))
            return res
        fsMap = {"4mu": "BsTo2DpTo4Mu", "2mu2e": "BsTo2DpTo2Mu2e"}
        dpMassMap = {0.25: "0p25", 1.2: "1p2", 5: "5"}

        mXX_ = []
        if mXX is None:
            mXX_ = self.fmXX
        elif isinstance(mXX, numbers.Number):
            if mXX not in self.fmXX:
                print('BoundState mass "{}" not available!'.format(mXX))
                print("Choose from {}".format(self.fmXX))
                return []
            mXX_ = [mXX]
        elif isinstance(mXX, list):
            for _ in mXX:
                if _ not in self.fmXX:
                    print('BoundState mass "{}" not available!'.format(_))
                    print("Choose from {}".format(self.fmXX))
                    return []
                mXX_.append(_)
        else:
            raise ValueError

        mA_ = []
        if mA is None:
            mA_ = self.fmA
        elif isinstance(mA, numbers.Number):
            if mA not in self.fmA:
                print('Darkphoton mass "{}" not available!'.format(mA))
                print("Choose from {}".format(self.fmA))
                return []
            mA_ = [mA]
        elif isinstance(mA, list):
            for _ in mA:
                if _ not in self.fmA:
                    print('Darkphoton mass "{}" not available!'.format(_))
                    print("Choose from {}".format(self.fmA))
                    return []
                mA_.append(_)
        else:
            raise ValueError
        mA_ = [dpMassMap[m] for m in mA_]

        dirs = getSampleDirInEOS(fsMap[fs], mXX_, mA_)
        files_ = {
            k: uch.getffNtuples(uch.getTimestampedDir(v, nth=nth))
            for k, v in dirs.items()
        }

        if grouped:
            res = files_
        else:
            res = [f for x in files_.values() for f in x]
        return res


def main():

    sd = SignalDescription()

    print("## final states", sd.finalstates, sep="\n")
    print("## mXXs", sd.mXX, sep="\n")
    print("## mAs", sd.mA, sep="\n")
    print('## files("4mu")', *sd.files("4mu"), sep="\n")
    print('## files("2mu2e")', *sd.files("2mu2e"), sep="\n")
    print(
        '## files("4mu", mXX=[100, 500], mA=0.25',
        *sd.files("4mu", mXX=[100, 500], mA=0.25),
        sep="\n"
    )


if __name__ == "__main__":
    main()
