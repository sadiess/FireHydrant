#!/usr/bin/env python
import utils.commonHelpers as uch

FFNTUPLE_ROOTDIR = "/store/group/lpcmetx/MCSIDM/ffNtuple/2018"

DATALIST = """\
/DoubleMuon/Run2018A-17Sep2018-v2/AOD
/DoubleMuon/Run2018B-17Sep2018-v1/AOD
/DoubleMuon/Run2018C-17Sep2018-v1/AOD
/DoubleMuon/Run2018D-PromptReco-v2/AOD""".split()

DATALIST = ["/DoubleMuon/Run2018C-17Sep2018-v1/AOD"] # for the moment

class DataInfo:
    def __init__(self, datasetName):
        self.fDatasetName = datasetName
        self.fNtuplePath = FFNTUPLE_ROOTDIR + "/".join(datasetName.split("/")[:-1])
        self.fNtupleFiles = []
        self.fNth = -1
        self.fEra = self.fDatasetName.split("/")[2].split("-")[0][-1]

    def __repr__(self):
        return f"Location: {self.fNtuplePath}\nFrom: {fDatasetName}"

    @property
    def era(self):
        return self.fEra

    def files(self, nth=-1, reload=False):
        if nth != self.fNth or reload or len(self.fNtupleFiles) == 0:
            try:
                self.fNtupleFiles = uch.getffNtuples(
                    uch.getTimestampedDir(self.fNtuplePath, nth=nth)
                )
                self.fNth = nth
            except:
                print("Path: {} DOES NOT EXIST IN EOS!".format(self.fNtuplePath))
                raise
        return self.fNtupleFiles


class DataDescription:
    def __init__(self):
        self.fDataStore = [DataInfo(d) for d in DATALIST]

    def eras(self):
        return [d.era for d in self.fDataStore]

    def files(self, era="*", nth=-1):
        if era in self.eras():
            return [d for d in self.fDataStore if d.era == era][0].files(nth=nth)
        elif era == "*":
            res = []
            for d in self.fDataStore:
                res.extend(d.files(nth=nth))
            return res
        else:
            print(f"DataFile for era {era} not available!")
            return []

    def filesByEra(self, nth=-1):
        return dict(zip(self.eras(), [d.files(nth=nth) for d in self.fDataStore]))


if __name__ == "__main__":
    dd = DataDescription()

    print(f"Eras: {dd.eras()}")
    print(f"Files: {dd.files()}")
    print(f"FilesByEra: {dd.filesByEra()}")
