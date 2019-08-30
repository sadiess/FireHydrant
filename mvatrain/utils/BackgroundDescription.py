#!/usr/bin/env python

from functools import reduce
from utils.backgroundList import datasetWithWeight
from utils.BackgroundInfo import BackgroundInfo


class BackgroundDescription:
    def __init__(self):
        self.fDatasetWithWeight = datasetWithWeight
        self.fInfos = [BackgroundInfo(ds) for ds in datasetWithWeight.keys()]

    @property
    def datasetWithWeight(self):
        return self.fDatasetWithWeight

    def types(self, validOnly=True):
        return set(
            [info_.type for info_ in self.fInfos if not validOnly or info_.isValid]
        )

    def tags(self, validOnly=True):
        return set(
            [info_.tag for info_ in self.fInfos if not validOnly or info_.isValid]
        )

    def datasets(self, validOnly=True):
        return [
            info_.dataset for info_ in self.fInfos if not validOnly or info_.isValid
        ]

    def lookupWeight(self, datasetName):
        try:
            return self.fDatasetWithWeight[datasetName]
        except KeyError as e:
            print("dataset: {} does NOT exist!".format(datasetName))
            print("Available list:", *list(self.fDatasetWithWeight.keys()), sep="\n")
            return -1

    def getDatasetsFromType(self, type, validOnly=True):
        return [
            info_.dataset
            for info_ in self.fInfos
            if info_.type == type and (not validOnly or info_.isValid)
        ]

    def getDatasetsFromTag(self, tag, validOnly=True):
        return [
            info_.dataset
            for info_ in self.fInfos
            if info_.tag == tag and (not validOnly or info_.isValid)
        ]

    def getFileWeightFromType(self, type, nth=-1):
        res = []
        for info_ in self.fInfos:
            if not info_.isValid:
                continue
            if info_.type != type:
                continue
            files_ = info_.files(nth=nth)
            if files_:
                res.append((files_, info_.weight))
        return res

    def getFileWeightFromTag(self, tag, nth=-1):
        res = []
        for info_ in self.fInfos:
            if not info_.isValid:
                continue
            if info_.tag != tag:
                continue
            files_ = info_.files(nth=nth)
            if files_:
                res.append((files_, info_.weight))
        return res

    def getTotalFileWeights(self, groupby=None, validOnly=True, nth=-1):
        if groupby is None:
            res_ = [
                (info_.files(nth=nth), info_.weight)
                for info_ in self.fInfos
                if (not validOnly or info_.isValid)
            ]
            return [r for r in res_ if r[0]]
        elif groupby == "type":
            return {
                t: self.getFileWeightFromType(t, nth=nth)
                for t in self.types(validOnly=validOnly)
            }
        elif groupby == "tag":
            return {
                t: self.getFileWeightFromTag(t, nth=nth)
                for t in self.tags(validOnly=validOnly)
            }
        else:
            raise ValueError("groupby can only be [None, 'type', 'tag']")

    def getTotalFiles(self, validOnly=True, nth=-1):
        return reduce(
            lambda x, y: x + y,
            [fw[0] for fw in self.getTotalFileWeights(validOnly=validOnly, nth=nth)],
        )


if __name__ == "__main__":
    bd = BackgroundDescription()

    for t in bd.tags():
        print(f"** {t} **")
        for ds in bd.getDatasetsFromTag(t):
            print("\t", ds)
        print()
