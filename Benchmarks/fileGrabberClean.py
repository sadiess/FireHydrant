#!/usr/bin/env python

import uproot
import json
import pyxrootd.client
import fnmatch
import numpy as np
import numexpr
import subprocess
import concurrent.futures
import warnings
import os
import difflib
from optparse import OptionParser
from process import *

parser = OptionParser()
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()
fnaleos = "root://cmsxrootd.fnal.gov/"

beans={}
beans['2018'] = ["/store/group/lpcmetx/MCSIDM/ffNtuple/2018"]

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs


xsections={}
for k in processes.items():
	xsections[k] = -1

datadef = {}
for folder in beans[options.year]:
    print("Opening",folder)
    for dataset in xsections.keys():
        if options.dataset and options.dataset not in dataset: continue 
        print("Looking into",folder+"/"+dataset)
        os.system("find "+folder+"/"+dataset+" -name \'*.root\' > "+dataset+".txt")
        flist = open(dataset+".txt")
        urllist = []
        #print('file lenght:',len(flist.readlines()))
        xs = xsections[dataset]
        #sumw = 0
        for path in flist:
            s = path.strip().split('/') #.strip() removes whitespace, .split() does the split defined earlier
            eospath = fnaleos
            for i in range (3,len(s)): eospath=eospath+'/'+s[i]
            #if xs != -1:
            #     run_tree = uproot.open(eospath)["Runs"]
            #     sumw += run_tree.array("genEventSumw")[0]
            if (not ('failed' in eospath)): urllist.append(eospath)
        print('list lenght:',len(urllist))
        urllists = split(urllist, int(options.pack))
        print(len(urllists))
        if urllist:
            for i in range(0,len(urllists)) : #this is the step that actually fills the dict defined above
                 datadef[dataset+"____"+str(i)] = {
                      'files': urllists[i],
                      'xs': xs,
                      #'sumw': sumw,
                      }
        os.system("rm "+dataset+".txt")

os.system("mkdir -p beans") #creates target directory for final json file to land in
with open("beans/"+options.year+".json", "w") as fout: #sets up formatting for file that goes in that directory
    json.dump(datadef, fout, indent=4)