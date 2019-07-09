#!/usr/bin/env python

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
from optparse import OptionParser
from process import *

parser = OptionParser()
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()
fnaleos = "root://cmseos.fnal.gov/"

beans={}
beans['2018'] = ['/store/group/lpcmetx/MCSIDM/ffNtuple/2018']

def split(arr, size):
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs


xsections={}
for k in processes.keys():
    xsections[k] = -1

datadef = {}
for folder in beans['2018']:
    print("Opening", folder)
    for dataset in xsections.keys():
        if options.dataset and options.dataset not in dataset: continue 
        print("Looking into", folder+"/"+dataset)
        totalpath = folder+"/"+dataset
        cmd = "eos {0} find -f --xurl {1} > {2}".format("root://cmseos.fnal.gov/", totalpath, dataset+".txt")
        os.system(cmd)
        flist = open(dataset+".txt")
        urllist = []
        f = flist.readlines()
        print('file length:',len(f))
        xs = xsections[dataset]
        for path in f:
            s = path.strip().split('/')
            eospath = fnaleos
            for i in range (0,len(s)): eospath=eospath+'/'+s[i]
            if (not ('failed' in eospath)): urllist.append(eospath)
        print('list length:', len(urllist))
        urllists = split(urllist, int(options.pack))
        if urllist:
            for i in range(0,len(urllists)) :
                 datadef[dataset+"____"+str(i)] = {
                      'files': urllists[i],
                      'xs': xs,
                      }
        os.system("rm "+dataset+".txt")

os.system("mkdir -p beans")
with open("beans/"+options.year+".json", "w") as fout:
    json.dump(datadef, fout, indent=4)