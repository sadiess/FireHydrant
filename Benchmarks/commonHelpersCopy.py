#!/usr/bin/env python

"""
from utils/commonHelpers from weinan's code
"""

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