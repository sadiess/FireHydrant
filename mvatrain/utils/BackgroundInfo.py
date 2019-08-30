#!/usr/bin/env python
import os
from utils.backgroundList import datasetWithWeight
import utils.commonHelpers as uch

FFNTUPLE_ROOTDIR = "/store/group/lpcmetx/MCSIDM/ffNtuple/2018"


class BackgroundInfo:
    def __init__(self, datasetName):
        self.fDatasetName = datasetName
        self.fValid = datasetName.startswith("/") and len(datasetName.split("/")) == 4
        self.fType = datasetName.split("/")[1].split("_")[0]
        self.fWeight = datasetWithWeight.get(datasetName, 0.0)
        self.fValid = self.fValid and bool(self.fWeight)
        self.fNtuplePath = FFNTUPLE_ROOTDIR + "/".join(datasetName.split("/")[:-1])
        self.fNtupleFiles = []
        self.fNth = -1

    def __str__(self):
        infostrs = [
            f"type: {self.fType}",
            f"dataset: {self.fDatasetName}",
            f"valid: {self.fValid}",
            f"weight: {self.fWeight}",
        ]
        return str(infostrs)

    def __repr__(self):
        fmt = "{:10}{}"
        infostrs = [
            fmt.format("type", self.fType),
            fmt.format("dataset", self.fDatasetName),
            fmt.format("valid", self.fValid),
            fmt.format("weight", self.fWeight),
        ]
        return "\n".join(infostrs)

    @property
    def isValid(self):
        return self.fValid

    @property
    def dataset(self):
        return self.fDatasetName

    @property
    def type(self):
        return self.fType

    @property
    def tag(self):
        res = self.fType
        if len(self.fType) == 3 and all(
            [x in list("WZG") for x in list(self.fType.upper())]
        ):
            res = "triboson"
        if len(self.fType) == 2 and all(
            [x in list("WZ") for x in list(self.fType.upper())]
        ):
            res = "diboson"
        return res

    @property
    def weight(self):
        return self.fWeight

    @property
    def batch(self):
        return self.fNth

    @property
    def ntuplePath(self):
        return self.fNtuplePath

    def files(self, nth=-1, reload=False):
        if nth != self.fNth or reload or len(self.fNtupleFiles) == 0:
            try:
                self.fNtupleFiles = uch.getffNtuples(
                    uch.getTimestampedDir(self.fNtuplePath, nth=nth)
                )
                self.fNth = nth
            except:
                print("Path: {} DOES NOT EXIST IN EOS!".format(self.fNtuplePath))
                self.fValid = False
                self.fNtupleFiles = []
        return self.fNtupleFiles
