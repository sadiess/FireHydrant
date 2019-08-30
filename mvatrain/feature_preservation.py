#!/usr/bin/env python3
import os
import time

import numpy as np
import awkward
import utils.histoHelpers as uhh
import utils.uprootHelpers as uuh
import mvatrain.preprocessors as mpp
from utils.SignalDescription import SignalDescription
from utils.BackgroundDescription import BackgroundDescription

DATA_DIR = os.path.join(os.environ["FFANA_BASE"], "mvatrain/data")
TIME_STR = time.strftime("%y%m%d")  # 190530

# picker functions-------------------------------------------------------------
def match(obj):
    t = obj.tree
    genp4 = uuh.p4Array(t["gen_p4"])
    mDarkPhoton = t["gen_pid"].array() == 32
    jetp4 = uuh.p4Array(t["pfjet_p4"])
    mGendpMatch, mJetMatch = uuh.MaskArraysFromMatching(genp4[mDarkPhoton], jetp4)
    res = mJetMatch[obj.mHLT].flatten()
    return res


def pt(obj):
    t = obj.tree
    return uuh.p4Array(t["pfjet_p4"])[obj.mHLT].pt.flatten()


def eta(obj):
    t = obj.tree
    return uuh.p4Array(t["pfjet_p4"])[obj.mHLT].eta.flatten()


def neutralEnergyFrac(obj):
    t = obj.tree
    jetp4 = uuh.p4Array(t["pfjet_p4"])
    _res = (
        t["pfjet_neutralEmE"].array() + t["pfjet_neutralHadronE"].array()
    ) / jetp4.energy
    return _res[obj.mHLT].flatten()


def pickExtreme(ja, maxmin):
    if ja.size == 0:
        return np.array([])
    if maxmin == "max":
        return np.array([max(x) if x.size else np.nan for x in ja])
    if maxmin == "min":
        return np.array([min(x) if x.size else np.nan for x in ja])
    return np.array([])


def maxd0(obj):
    t = obj.tree
    candTkd0 = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_tkD0"].array())[
        obj.mHLT
    ].flatten()
    return pickExtreme(candTkd0, "max")


def mind0(obj):
    t = obj.tree
    candTkd0 = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_tkD0"].array())[
        obj.mHLT
    ].flatten()
    return pickExtreme(candTkd0, "min")


def n_muon(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    return (np.abs(candType) == 3).sum()


def n_dsa(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    return (np.abs(candType) == 8).sum()


def n_electron(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    return (np.abs(candType) == 2).sum()


def n_photon(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    return (np.abs(candType) == 4).sum()


def energy_muon(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    candEnergy = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_energy"].array())[
        obj.mHLT
    ].flatten()

    return candEnergy[np.abs(candType) == 3].sum()


def energy_dsa(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    candEnergy = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_energy"].array())[
        obj.mHLT
    ].flatten()

    return candEnergy[np.abs(candType) == 8].sum()


def energy_electron(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    candEnergy = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_energy"].array())[
        obj.mHLT
    ].flatten()

    return candEnergy[np.abs(candType) == 2].sum()


def energy_photon(obj):
    t = obj.tree
    candType = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_type"].array())[
        obj.mHLT
    ].flatten()
    candEnergy = uuh.NestNestObjArrayToJagged(t["pfjet_pfcand_energy"].array())[
        obj.mHLT
    ].flatten()

    return candEnergy[np.abs(candType) == 4].sum()


def tkiso(obj):
    return obj.tree["pfjet_tkIsolation05"].array()[obj.mHLT].flatten()


def pfiso(obj):
    return obj.tree["pfjet_pfIsolation05"].array()[obj.mHLT].flatten()


def ptspread(obj):
    return obj.tree["pfjet_ptDistribution"].array()[obj.mHLT].flatten()


def drspread(obj):
    return obj.tree["pfjet_dRSpread"].array()[obj.mHLT].flatten()


def jetsub_lambda(obj):
    return obj.tree["pfjet_subjet_lambda"].array()[obj.mHLT].flatten()


def jetsub_epsilon(obj):
    return obj.tree["pfjet_subjet_epsilon"].array()[obj.mHLT].flatten()


def ecf1(obj):
    return obj.tree["pfjet_subjet_ecf1"].array()[obj.mHLT].flatten()


def ecf2(obj):
    return obj.tree["pfjet_subjet_ecf2"].array()[obj.mHLT].flatten()


def ecf3(obj):
    return obj.tree["pfjet_subjet_ecf3"].array()[obj.mHLT].flatten()


# -----------------------------------------------------------------------------


_pm = {
    "target": match,
    "pt": pt,
    "eta": eta,
    "neufrac": neutralEnergyFrac,
    "maxd0": maxd0,
    "mind0": mind0,
    # "nmuo": n_muon,
    # "ndsa": n_dsa,
    # "nele": n_electron,
    # "npho": n_photon,
    # "energymuo": energy_muon,
    # "energydsa": energy_dsa,
    # "energyele": energy_electron,
    # "energypho": energy_photon,
    "tkiso": tkiso,
    "pfiso": pfiso,
    "spreadpt": ptspread,
    "spreaddr": drspread,
    "lambda": jetsub_lambda,
    "epsilon": jetsub_epsilon,
    "ecf1": ecf1,
    "ecf2": ecf2,
    "ecf3": ecf3,
}

# -----------------------------------------------------------------------------


def preserve_signal():
    print("Getting signal samples from EOS...", end=" ")
    sd = SignalDescription()
    sig4mu = sd.files("4mu", grouped=True, nth=-1)
    sig2mu2e = sd.files("2mu2e", grouped=True, nth=-1)
    sigfs4mu = [f for x in sig4mu.values() for f in x]
    sigfs2mu2e = [f for x in sig2mu2e.values() for f in x]
    print(
        "Done!\nNumber of files - (4mu: {} | 2mu2e: {})".format(
            len(sigfs4mu), len(sigfs2mu2e)
        )
    )

    mp = mpp.ffMultiPicker(sigfs4mu + sigfs2mu2e, pickmethods=_pm)
    mp_res = mp.pick()

    sigfn = "signal_{}.awkd".format(TIME_STR)
    sigfn = os.path.join(DATA_DIR, sigfn)
    awkward.save(sigfn, mp_res, mode="w")
    print("Signal sample labels:", np.unique(mp_res["target"], return_counts=True))

    return sigfn


def preserve_background():
    print("Getting background samples from EOS...", end=" ")
    bd = BackgroundDescription()
    bkgfs = bd.getTotalFiles()
    print("Done!\nNumber Of files - bkg: {}".format(len(bkgfs)))

    _pm.pop("target")  # remove label, no gen info for backgrounds
    mp = mpp.ffMultiPicker(filenames=bkgfs, pickmethods=_pm)
    mp_res = mp.pick()
    print("Background sample size:", mp_res["pt"].size)

    # make up target
    mp_res["target"] = np.zeros(len(mp_res["pt"]), dtype=bool)

    bkgfn = "bkg_{}.awkd".format(TIME_STR)
    bkgfn = os.path.join(DATA_DIR, bkgfn)
    awkward.save(bkgfn, mp_res, mode="w")

    return bkgfn


def preservation_combine(sigfn, bkgfn):
    sig = awkward.load(sigfn)
    bkg = awkward.load(bkgfn)

    combo_ = dict()
    for k in sig.keys():
        combo_[k] = np.concatenate([sig[k], bkg[k]])
    print("Combined sample size:", combo_["pt"].size)
    combfn = "combo_{}.awkd".format(TIME_STR)
    combfn = os.path.join(DATA_DIR, combfn)
    awkward.save(combfn, combo_, mode="w")
    print("Combined sample preserved at:", combfn)
    os.remove(sigfn)
    os.remove(bkgfn)


if __name__ == "__main__":

    sigfn = preserve_signal()
    bkgfn = preserve_background()
    preservation_combine(sigfn, bkgfn)
