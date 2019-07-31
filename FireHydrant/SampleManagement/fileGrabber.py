#!/usr/bin/env python

'''
When given a starter folder (see beans['2018']) and a list of dataset subfolders (found in process.py),
this script will output a .json file containing complete filepaths to all .root files, as well as the
weights associated with those files. Signal and data files are assigned a weight of 1.
'''

import uproot
import json
import pyxrootd.client
import fnmatch
import numpy as np
import subprocess
import concurrent.futures
import warnings
import os
import difflib
import re
from optparse import OptionParser
from datetime import datetime
from process import *
from eventCounter import *

parser = OptionParser()
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()

beans={}
beans['2018'] = ['/store/group/lpcmetx/MCSIDM/ffNtuple/2018']
beans['TTJets'] = ['/store/group/lpcmetx/MCSIDM/ffNtuple/2018', '/store/group/lpcmetx/SIDM/ffNtuple/2018']
beans['Signal'] = ['/store/group/lpcmetx/MCSIDM/ffNtuple/2018', '/store/group/lpcmetx/SIDM/ffNtuple/2018']

datadataset = dict.fromkeys(['Run2018A-17Sep2018-v2', 'Run2018B-17Sep2018-v1', 'Run2018C-17Sep2018-v1', 'Run2018D-PromptReco-v2'])
signaldataset = dict.fromkeys(['SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-0p25_ctau-1p875', 'SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-0p8_ctau-6', 'SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-1p2_ctau-9', 'SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-2p5_ctau-18p75', 'SIDM_BsTo2DpTo2Mu2e_MBs-1000_MDp-5_ctau-37p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-0p25_ctau-18p75', 'SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-0p8_ctau-60', 'SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-1p2_ctau-90', 'SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-2p5_ctau-187p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-100_MDp-5_ctau-375', 'SIDM_BsTo2DpTo2Mu2e_MBs-150_MDp-0p25_ctau-12p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-150_MDp-0p8_ctau-40', 'SIDM_BsTo2DpTo2Mu2e_MBs-150_MDp-1p2_ctau-60', 'SIDM_BsTo2DpTo2Mu2e_MBs-150_MDp-2p5_ctau-125', 'SIDM_BsTo2DpTo2Mu2e_MBs-150_MDp-5_ctau-250', 'SIDM_BsTo2DpTo2Mu2e_MBs-200_MDp-0p25_ctau-9p375', 'SIDM_BsTo2DpTo2Mu2e_MBs-200_MDp-0p8_ctau-30', 'SIDM_BsTo2DpTo2Mu2e_MBs-200_MDp-1p2_ctau-45', 'SIDM_BsTo2DpTo2Mu2e_MBs-200_MDp-2p5_ctau-93p75', 'SIDM_BsTo2DpTo2Mu2e_MBs-200_MDp-5_ctau-187p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-500_MDp-0p25_ctau-3p75', 'SIDM_BsTo2DpTo2Mu2e_MBs-500_MDp-0p8_ctau-12', 'SIDM_BsTo2DpTo2Mu2e_MBs-500_MDp-1p2_ctau-18', 'SIDM_BsTo2DpTo2Mu2e_MBs-500_MDp-2p5_ctau-37p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-500_MDp-5_ctau-75','SIDM_BsTo2DpTo2Mu2e_MBs-800_MDp-0p25_ctau-2p344', 'SIDM_BsTo2DpTo2Mu2e_MBs-800_MDp-0p8_ctau-7p5', 'SIDM_BsTo2DpTo2Mu2e_MBs-800_MDp-1p2_ctau-11p25', 'SIDM_BsTo2DpTo2Mu2e_MBs-800_MDp-2p5_ctau-23p438', 'SIDM_BsTo2DpTo2Mu2e_MBs-800_MDp-5_ctau-46p875', 'SIDM_BsTo2DpTo4Mu_MBs-1000_MDp-0p25_ctau-1p875', 'SIDM_BsTo2DpTo4Mu_MBs-1000_MDp-0p8_ctau-6', 'SIDM_BsTo2DpTo4Mu_MBs-1000_MDp-1p2_ctau-9', 'SIDM_BsTo2DpTo4Mu_MBs-1000_MDp-2p5_ctau-18p75', 'SIDM_BsTo2DpTo4Mu_MBs-1000_MDp-5_ctau-37p5', 'SIDM_BsTo2DpTo4Mu_MBs-100_MDp-0p25_ctau-18p75', 'SIDM_BsTo2DpTo4Mu_MBs-100_MDp-0p8_ctau-60', 'SIDM_BsTo2DpTo4Mu_MBs-100_MDp-1p2_ctau-90', 'SIDM_BsTo2DpTo4Mu_MBs-100_MDp-2p5_ctau-187p5', 'SIDM_BsTo2DpTo4Mu_MBs-100_MDp-5_ctau-375', 'SIDM_BsTo2DpTo4Mu_MBs-150_MDp-0p25_ctau-12p5', 'SIDM_BsTo2DpTo4Mu_MBs-150_MDp-0p8_ctau-40', 'SIDM_BsTo2DpTo4Mu_MBs-150_MDp-1p2_ctau-60', 'SIDM_BsTo2DpTo4Mu_MBs-150_MDp-2p5_ctau-125', 'SIDM_BsTo2DpTo4Mu_MBs-150_MDp-5_ctau-250', 'SIDM_BsTo2DpTo4Mu_MBs-200_MDp-0p25_ctau-9p375', 'SIDM_BsTo2DpTo4Mu_MBs-200_MDp-0p8_ctau-30', 'SIDM_BsTo2DpTo4Mu_MBs-200_MDp-1p2_ctau-45', 'SIDM_BsTo2DpTo4Mu_MBs-200_MDp-2p5_ctau-93p75', 'SIDM_BsTo2DpTo4Mu_MBs-200_MDp-5_ctau-187p5', 'SIDM_BsTo2DpTo4Mu_MBs-500_MDp-0p25_ctau-3p75', 'SIDM_BsTo2DpTo4Mu_MBs-500_MDp-0p8_ctau-12', 'SIDM_BsTo2DpTo4Mu_MBs-500_MDp-1p2_ctau-18', 'SIDM_BsTo2DpTo4Mu_MBs-500_MDp-2p5_ctau-37p5', 'SIDM_BsTo2DpTo4Mu_MBs-500_MDp-5_ctau-75', 'SIDM_BsTo2DpTo4Mu_MBs-800_MDp-0p25_ctau-2p344', 'SIDM_BsTo2DpTo4Mu_MBs-800_MDp-0p8_ctau-7p5', 'SIDM_BsTo2DpTo4Mu_MBs-800_MDp-1p2_ctau-11p25', 'SIDM_BsTo2DpTo4Mu_MBs-800_MDp-2p5_ctau-23p438', 'SIDM_BsTo2DpTo4Mu_MBs-800_MDp-5_ctau-46p875'])

lumi = 60.432*1e3 #This is the luminosity of the desired period for weight scaling

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs


xsections={}
for k,v in processes.items():
    if v[0]=='MC':
        xsections[k] = v[1]
    else:
        xsections[k] = 1

datadef = {}
for folder in beans['2018']:
    print("Opening", folder)
    for dataset in xsections.keys():
        if options.dataset and options.dataset not in dataset: continue 
            
        timestampdirs = []
        urllist = []
        
        '''
        The Dataset 'QCD_Pt-30to50' is not available on disk. This section masks out this dataset, but
        can be removed if the dataset becomes available.
        '''

        if dataset == 'QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8':
            print("Looking into", dataset)
            print('This dataset has been masked out.')
            continue

        '''
        The TTJets dataset has data stored in two different places, so we check both for the most recent
        timestamp.
        '''
            
        elif dataset == 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8':
            latest = []
            for extrafolder in beans['TTJets']:
                print("Opening", extrafolder)
                print("Looking into", extrafolder+"/"+dataset)
                totalpath = extrafolder+"/"+dataset
                timestampdirs = []
                cmd = "eos {0} find -f {1} | grep 'ffNtuple' > {2}".format("root://cmseos.fnal.gov/", totalpath, dataset+"_subdirs.txt")
                os.system(cmd)
                slist=open(dataset+"_subdirs.txt")
                s = slist.readlines()
                for apath in s:
                    if "/failed/" in apath:
                        continue
                    splits = apath.strip().split('/')
                    for i in range (0, (len(splits)-1)):
                        if splits[i] == '':
                            splits.pop(i)
                    if extrafolder == '/store/group/lpcmetx/MCSIDM/ffNtuple/2018':
                        timestamp = splits[-3]
                    else:
                        timestamp = splits[-2]
                    timestampdirs.append(timestamp)
                
                timestampdirs = sorted(
                    timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                )

                latest.append(timestampdirs[-1])
                
            truelatest = sorted(
                latest, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
            )
            
            truelatest = truelatest[-1]
            
            for extrafolder in beans['TTJets']:
                    testpath = extrafolder+"/"+dataset
                    cmd = "eos {0} find -d {1} | grep {2} > test.txt".format("root://cmseos.fnal.gov/", testpath, truelatest)
                    os.system(cmd)
                    tlist = open("test.txt")
                    t = tlist.readlines()
                    for apath in t:
                        if truelatest in apath:
                            totalpath = testpath
                            os.system("rm test.txt")
                        else:
                            os.system("rm test.txt")
            print("totalpath: ", totalpath)
            cmd = "eos {0} find -f --xurl {1} | grep {2} > {3}".format("root://cmseos.fnal.gov/", totalpath, truelatest, dataset+".txt")
            os.system(cmd)
            flist = open(dataset+".txt")
            f = flist.readlines()
            print('file length:',len(f))
            
        '''
        The DoubleMuon dataset has four subsections, A, B, C, and D. We want the most recent dataset for
        each, not overall.
        '''

        elif dataset == 'DoubleMuon':
            print("Looking into", folder+"/"+dataset)
            totalpath = folder+"/"+dataset
            f = []
            latest = []

            cmd = "eos {0} find -d {1} | grep '000*/$' > {2}".format("root://cmseos.fnal.gov/", totalpath, dataset+"_subdirs.txt")
            os.system(cmd)
            slist=open(dataset+"_subdirs.txt")
            s = slist.readlines()
            for abcd in datadataset.keys():
                print('looking at: ', abcd)
                timestampdirs = []
                for apath in s:
                    if "/failed/" in apath:
                        continue
                    if not (abcd in apath):
                        continue
                    if '0000' in apath or '0001' in apath:
                        splits = apath.strip().split('/')
                        for i in range(0, len(splits)):
                            if splits[i] == '':
                                splits.pop(i)
                        timestamp = splits[-2]
                        timestampdirs.append(timestamp)

                timestampdirs = sorted(
                    timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                )

                latest = timestampdirs[-1]
            
                cmd = "eos {0} find -f --xurl {1} | grep {2} > {3}".format("root://cmseos.fnal.gov/", totalpath, latest,        abcd+'_'+dataset+".txt")
                os.system(cmd)
                flist = open(abcd+'_'+dataset+".txt")
                f += flist.readlines()
                print('file length:', len(f))
                os.system("rm "+abcd+'_'+dataset+".txt")
                
        '''
        The CRAB_PrivateMC dataset has data stored in two different places, as well as subsections like DoubleMuon does.
        We check both places for the most recent dataset, and store for each subsection.
        '''

        elif dataset == 'CRAB_PrivateMC':
            f = []
            latest = []
            for ctau in signaldataset.keys():
                print("Looking at: ", ctau)
                timestampdirs = []
                latest = []
                for extrafolder in beans['Signal']:
                    print('Opening ', extrafolder)
                    print("Looking into", extrafolder+"/"+dataset)
                    totalpath = extrafolder+"/"+dataset
                    cmd = "eos {0} find -d {1} | grep '000*/$' > {2}".format("root://cmseos.fnal.gov/", totalpath, dataset+"_subdirs.txt")
                    os.system(cmd)
                    slist=open(dataset+"_subdirs.txt")
                    s = slist.readlines()
                    for apath in s:
                        if "/failed/" in apath:
                            continue
                        if not (ctau in apath):
                            continue
                        if '0000' in apath or '0001' in apath:
                            splits = apath.strip().split('/')
                            for i in range(0, len(splits)):
                                if splits[i] == '':
                                    splits.pop(i)
                            timestamp = splits[-2]
                            timestampdirs.append(timestamp)

                    timestampdirs = sorted(
                        timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                    )

                    latest.append(timestampdirs[-1])
                    
                truelatest = sorted(
                    latest, key=lambda x: datetime. strptime(x, "%y%m%d_%H%M%S")
                )
                
                truelatest = truelatest[-1]
                
                for extrafolder in beans['Signal']:
                    testpath = extrafolder+"/"+dataset
                    cmd = "eos {0} find -d {1} | grep {2} > test.txt".format("root://cmseos.fnal.gov/", testpath, truelatest)
                    os.system(cmd)
                    tlist = open("test.txt")
                    t = tlist.readlines()
                    for apath in t:
                        if truelatest in apath:
                            totalpath = testpath
                            os.system("rm test.txt")
                        else:
                            os.system("rm test.txt")
                
                cmd = "eos {0} find -f --xurl {1} | grep {2} > {3}".format("root://cmseos.fnal.gov/", totalpath, truelatest,        ctau+'_'+dataset+".txt")
                os.system(cmd)
                flist = open(ctau+'_'+dataset+".txt")
                f += flist.readlines()
                print('file length:', len(f))
                os.system("rm "+ctau+'_'+dataset+".txt")

        '''
        The rest of the datasets have normal data storage.
        '''
                
        else:
            print("Looking into", folder+"/"+dataset)
            totalpath = folder+"/"+dataset
            latest = []

            cmd = "eos {0} find -d {1} | grep '000*/$' > {2}".format("root://cmseos.fnal.gov/", totalpath, dataset+"_subdirs.txt")
            os.system(cmd)
            slist=open(dataset+"_subdirs.txt")
            s = slist.readlines()
            for apath in s:
                if "/failed/" in apath:
                    continue
                if '0000' in apath or '0001' in apath:
                    splits = apath.strip().split('/')
                    for i in range(0, len(splits)):
                        if splits[i] == '':
                            splits.pop(i)
                    timestamp = splits[-2]
                    timestampdirs.append(timestamp)

            timestampdirs = sorted(
                timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
            )

            latest = timestampdirs[-1]

            cmd = "eos {0} find -f --xurl {1} | grep {2} > {3}".format("root://cmseos.fnal.gov/", totalpath, latest, dataset+".txt")
            os.system(cmd)
            flist = open(dataset+".txt")
            f = flist.readlines()
            print('file length:',len(f))

        '''
        Weight is calculated from luminosity and cross-section.
        '''

        xs = xsections[dataset]
        for path in f:
            s = path.strip().split('/')
            eospath = ""
            for i in range (0,len(s)): eospath=eospath+'/'+s[i]
            eospath=eospath.strip('/')
            if (not ('failed' in eospath)):
                urllist.append(eospath)
                if xs != 1: #xs for data and signal events is assigned as 1
                    eventcount += processed_event_number(eospath)
                else:
                    eventcount = xs * lumi #to give a weight of 1 for data and signal events
                    
        scale = xs / eventcount
        weight = scale * lumi

        '''
        Finally, files, weight, and cross-section are all stored, and dumped into a single .json file.
        '''

        print('list length:', len(urllist))
        urllist = split(urllist, int(options.pack))
        if urllist:
            for i in range(0,len(urllist)) :
                 datadef[dataset+"____"+str(i)] = {
                      'files': urllist[i],
                      'weight': weight,
                      'xs': xs,
                      }
        os.system("rm "+dataset+".txt")
        os.system("rm "+dataset+"_subdirs.txt")

os.system("mkdir -p beans")
with open("beans/"+options.year+".json", "w") as fout:
    json.dump(datadef, fout, indent=4)

    
print("File successfully created.")