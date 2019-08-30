#!/usr/bin/env python
import concurrent.futures
import multiprocessing
import numpy as np
import utils.histoHelpers as uhh


class ffPicker(uhh.ffLoader):
    def __init__(self, filename, pickmethods={}):
        super().__init__(filename)
        self["methods"] = pickmethods

    @property
    def methods(self):
        return self["methods"]

    def setMethods(self, pickmethods):
        assert isinstance(pickmethods, dict)
        self["methods"] = pickmethods

    def updateMethods(self, pickmethods):
        assert isinstance(pickmethods, dict)
        self["methods"].update(pickmethods)

    def pick(self):
        res = {}
        for name, pm in self["methods"].items():
            res[name] = pm(self)
        return res


def singlePick(fn, methods):
    return ffPicker(fn, methods).pick()


class ffMultiPicker:
    def __init__(self, filenames, pickmethods={}):
        self._filenames = filenames
        if isinstance(filenames, str):
            self._filenames = [filenames]
        self._methods = pickmethods

    @property
    def methods(self):
        return self._methods

    @property
    def fileNames(self):
        return self._filenames

    def setMethods(self, pickmethods):
        assert isinstance(pickmethods, dict)
        self._methods = pickmethods

    def updateMethods(self, pickmethods):
        assert isinstance(pickmethods, dict)
        self._methods.update(pickmethods)

    def pick(self, strategy=concurrent.futures.ProcessPoolExecutor):
        if strategy not in (
            concurrent.futures.ProcessPoolExecutor,
            concurrent.futures.ThreadPoolExecutor,
        ):
            raise ValueError(
                """
                strategy can only be <concurrent.futures.ProcessPoolExecutor>
                or <concurrent.futures.ThreadPoolExecutor>
                """
            )
        res = {}
        with strategy(
            max_workers=min(len(self.fileNames), int(multiprocessing.cpu_count() * 0.8))
        ) as executor:
            futures = {
                executor.submit(singlePick, f, self.methods): f for f in self.fileNames
            }
            pickresults = []
            for future in concurrent.futures.as_completed(futures):
                fn = futures[future]
                try:
                    pickresults.append(future.result())
                except Exception as e:
                    print("=========================================")
                    print("Exception occured in ffMultiPicker.pick()")
                    print("File: ", fn)
                    print("Msg: ", str(e))
                    print("=========================================")

            if pickresults:
                for k in pickresults[0].keys():
                    res[k] = np.concatenate([r[k] for r in pickresults])
        return res


def multiPick(fns, methods):
    return ffMultiPicker(fns, methods).pick()
