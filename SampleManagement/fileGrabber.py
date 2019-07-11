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
beans['TTJets'] = '/store/group/lpcmetx/SIDM/ffNtuple/2018' #Needed for masking later, see line 68

lumi = 60.432 #This is the luminosity of the desired period for weight scaling

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
        
        '''
        The Dataset 'QCD_Pt-30to50' is not available on disk. Lines 66-70 mask out this dataset, but
        can be removed if the dataset becomes available.
        '''

        if dataset == 'QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8':
            print("Opening", extrafolder)
            print("Looking into", extrafolder+"/"+dataset)
            print('This dataset has been masked out.')
            continue
        
        '''
        The dataset 'TTJets' has files stored in 2 locations: /MCSIDM and /SIDM. This extra section,
        lines 80-96, checks the additional folder, then later the most recent timestamp between the two
        folders is selected.
        '''

        if dataset == 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8':
            extrafolder = beans['TTJets']
            print("Opening", extrafolder)
            print("Looking into", extrafolder+"/"+dataset)
            totalpath = extrafolder+"/"+dataset
            cmd = "eos {0} find -f {1} | grep 'ffNtuple' > {2}".format("root://cmseos.fnal.gov", totalpath, dataset+"_subdirs.txt")
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
                    timestamp = splits [-2]
                    timestampdirs.append(timestamp)
            
        print("Looking into", folder+"/"+dataset)
        totalpath = folder+"/"+dataset
        
        cmd = "eos {0} find -d {1} | grep '000*/$' > {2}".format("root://cmseos.fnal.gov", totalpath, dataset+"_subdirs.txt")
        os.system(cmd)
        slist=open(dataset+"_subdirs.txt")
        s = slist.readlines()
        
        for apath in s:
            if "/failed/" in apath:
                continue #filter out failed files
            if '0000' in apath or '0001' in apath: #ensures folders returned by the grep are at the correct level
                splits = apath.strip().split('/')
                for i in range(0, len(splits)):
                    if splits[i] == '': #fixes an issue where some lists added an empty string at the end after being split
                        splits.pop(i)
                timestamp = splits[-2]
                timestampdirs.append(timestamp)
                
        timestampdirs = sorted(
            timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
        )
        
        latest = timestampdirs[-1] #selects only most recent timestamp
        
        if dataset == 'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8':
            totalpath = extrafolder+'/'+dataset
        
        cmd = "eos {0} find -f --xurl {1} | grep {2} > {3}".format("root://cmseos.fnal.gov/", totalpath, latest, dataset+".txt")
        os.system(cmd)
        flist = open(dataset+".txt")
        f = flist.readlines()
        urllist = []
        weightlist = []
        
        print('file length:',len(f))
        xs = xsections[dataset]
        for path in f:
            s = path.strip().split('/')
            eospath = ""
            for i in range (0,len(s)): eospath=eospath+'/'+s[i]
            eospath=eospath.strip('/')
            if (not ('failed' in eospath)):
                urllist.append(eospath)
                if xs != 1: #weight for data and signal events is assigned as 1
                    eventcount = processed_event_number(eospath)
                    scale = xs / eventcount
                    weight = scale * lumi
                else:
                    weight = 1
                weightlist.append(weight)

        print('list length:', len(urllist))
        urllist = split(urllist, int(options.pack))
        weightlist = split(weightlist, int(options.pack))
        if urllist:
            for i in range(0,len(urllist)) :
                 datadef[dataset+"____"+str(i)] = {
                      'files and weights': (urllist[i], weightlist[i]),
                      'xs': xs,
                      }
        os.system("rm "+dataset+".txt")
        os.system("rm "+dataset+"_subdirs.txt")

os.system("mkdir -p beans")
with open("beans/"+options.year+".json", "w") as fout:
    json.dump(datadef, fout, indent=4)
    
print("File successfully created.")