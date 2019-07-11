#!/usr/bin/env python

import json
from datetime import datetime
from os.path import join
import uproot
import concurrent.futures

def processed_event_number(ntuplefile):
    """Given a ntuplefile path, return the number of events it ran over."""

    f_ = uproot.open(ntuplefile)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"history"))[0]
    
    return f_[key_].values[2]  # 0: run, 1: lumi, 2: events


def total_event_number(filelist):
    """Given a list of ntuple files, return the total number of events processed"""

    numevents = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(processed_event_number, f): f for f in filelist}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                numevents += future.result()
            except Exception as e:
                print(f">> Fail to get numEvents for {filename}\n{str(e)}")
    return numevents

