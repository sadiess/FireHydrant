#!/usr/bin/env python
"""Generate MC sample json file, until proper sample management tool show up.
"""
import json
from datetime import datetime
from os.path import join
import uproot
import concurrent.futures
from FireHydrant.Tools.commonhelpers import eosls, eosfindfile

EOSPATHS_BKG = dict(
    TTJets={
        "TTJets": [
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ]
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"
        ],
        "DYJetsToLL_M-50": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ],
    },
    DiBoson={
        "WW": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WW_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"
        ],
        "ZZ": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/ZZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"
        ],
        "WZ": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3"
        ],
    },
    TriBoson={
        "WWW": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2"
        ],
        "WWZ": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WWZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2"
        ],
        "WZZ": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2"
        ],
        "ZZZ": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2"
        ],
        "WZG": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WZG_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ],
        "WWG": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WWG_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2"
        ],
        "WGG": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/WGG_5f_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ],
    },
    QCD={
        "QCD_Pt-15to20": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3"
        ],
        "QCD_Pt-20to30": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v4"
        ],
        # "QCD_Pt-30to50": [
        #     "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3"
        # ],
        "QCD_Pt-50to80": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3"
        ],
        "QCD_Pt-80to120": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-120to170": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-170to300": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3"
        ],
        "QCD_Pt-300to470": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v1",
        ],
        "QCD_Pt-470to600": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-600to800": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ],
        "QCD_Pt-800to1000": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v2"
        ],
        "QCD_Pt-1000toInf": [
            "/store/group/lpcmetx/MCSIDM/ffNtuple/2018/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"
        ],
    },
)


BKG_XSEC = dict(
    TTJets={"TTJets": 491},
    DYJetsToLL={"DYJetsToLL-M-10to50": 15820, "DYJetsToLL_M-50": 5317},
    DiBoson={"WW": 75.91, "ZZ": 12.14, "WZ": 27.55},
    TriBoson={
        "WWW": 0.2154,
        "WWZ": 0.1676,
        "WZZ": 0.05701,
        "ZZZ": 0.01473,
        "WZG": 0.04345,
        "WWG": 0.2316,
        "WGG": 2.001,
    },
    QCD={
        "QCD_Pt-15to20": 2805000,
        "QCD_Pt-20to30": 2536000,
        # "QCD_Pt-30to50": 1375000,
        "QCD_Pt-50to80": 377900,
        "QCD_Pt-80to120": 89730,
        "QCD_Pt-120to170": 21410,
        "QCD_Pt-170to300": 7022,
        "QCD_Pt-300to470": 619.8,
        "QCD_Pt-470to600": 59.32,
        "QCD_Pt-600to800": 18.19,
        "QCD_Pt-800to1000": 3.271,
        "QCD_Pt-1000toInf": 1.08,
    },
)


def generate_background_json():

    generated = dict()
    for group in EOSPATHS_BKG:
        generated[group] = {}
        for tag in EOSPATHS_BKG[group]:
            generated[group][tag] = []
            for path in EOSPATHS_BKG[group][tag]:
                timestampdirs = eosls(path)
                timestampdirs = sorted(
                    timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                )
                latest = join(path, timestampdirs[-1])
                for filepath in eosfindfile(latest):
                    if "/failed/" in filepath:
                        continue  # filter out those in *failed* folder
                    generated[group][tag].append(filepath)

    with open("backgrounds.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


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


def generate_background_scale():
    """parse all files to get number of events processed => scale
        scale = xsec/#events, scale*lumi-> weight
    """

    bkgfilelist = json.load(open("backgrounds.json"))
    generated = dict()
    for group in BKG_XSEC:
        generated[group] = {}
        for tag in BKG_XSEC[group]:
            xsec = BKG_XSEC[group][tag]
            numevents = total_event_number(bkgfilelist[group][tag])
            generated[group][tag] = xsec / numevents

    with open("backgrounds_scale.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


def remove_empty_file(filepath):
    """given a file, if the tree has non-zero number of events, return filepath"""
    f_ = uproot.open(filepath)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"ffNtuple"))[0]
    if uproot.open(filepath)[key_].numentries != 0:
        return filepath
    else:
        return None


def remove_empty_files(filelist):
    """given a list of files, return all files with a tree of non-zero number of events"""
    cleanlist = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(remove_empty_file, f): f for f in filelist}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                res = future.result()
                if res:
                    cleanlist.append(res)
            except Exception as e:
                print(f">> Fail to get numEvents for {filename}\n{str(e)}")
    return cleanlist


def clean_background_json():
    """parse all background files, remove empty tree files
    """
    bkgfilelist = json.load(open("backgrounds.json"))
    for group in bkgfilelist:
        for tag in bkgfilelist[group]:
            files = bkgfilelist[group][tag]
            bkgfilelist[group][tag] = remove_empty_files(files)
    with open("backgrounds_nonempty.json", "w") as outf:
        outf.write(json.dumps(bkgfilelist, indent=4))


if __name__ == "__main__":

    ## Here we are only keeping the most recent submission batch
    generate_background_json()
    generate_background_scale()
    clean_background_json()
