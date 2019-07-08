#!/usr/bin/env python

"""
this is an attempt to combine decaf's Pack function and the getNtupleFiles method from Weinan's utils/commonHelpers
into a single function to read a long list of files into a single, usable json file that can then be read by
coffea plotting functions.
"""

import os
import sys
import getpass
import subprocess
import uproot
import json
import pyxrootd.client
import fnmatch
import numexpr
import subprocess
import concurrent.futures
import warnings
import os
import difflib
import numpy as np
from optparse import OptionParser
from process import *
from datetime import datetime
from functools import partial
from collections import namedtuple
from inspect import signature

sample = namedtuple("sample", "mxx mdp ctau proc") #for Weinan code later

parser = OptionParser() #creates new parser of command lines
parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
parser.add_option('-y', '--year', help='year', dest='year')
parser.add_option('-p', '--pack', help='pack', dest='pack')
(options, args) = parser.parse_args()
fnaleos = "root://cmsxrootd.fnal.gov/"

beans={} #this set up a dict of file directories. since our files are all under 2016 and 2018,
         #those are the only keys that are relevant. work just with 2018 for now though
#beans['2016'] = ["/store/group/lpcmetx/MCSIDM/ffNtuple/2016"]
beans['2018'] = ["/store/group/lpcmetx/MCSIDM/ffNtuple/2018"]

def split(arr, size): #this split function is used later
     arrs = []
     while len(arr) > size:
         pice = arr[:size]
         arrs.append(pice)
         arr   = arr[size:]
     arrs.append(arr)
     return arrs

'''

def parse_xsec(cfgfile): #still not sure why this exists, since the only instance I see of use is commented out
    xsec_dict = {}
    with open(cfgfile) as f:
        for l in f:
            l = l.strip()
            if l.startswith('#'):
                continue
            pieces = l.split()
            samp = None
            xsec = None
            isData = False
            for s in pieces:
                if 'AOD' in s:
                    samp = s.split('/')[1]
                    if 'AODSIM' not in s:
                        isData = True
                        break
                else:
                    try:
                        xsec = float(s)
                    except ValueError:
                        try:
                            import numexpr
                            xsec = numexpr.evaluate(s).item()
                        except:
                            pass
            if samp is None:
                print('Ignore line:\n%s' % l)
            elif not isData and xsec is None:
                print('Cannot find cross section:\n%s' % l)
            else:
                xsec_dict[samp] = xsec
    return xsec_dict

#xsections = parse_xsec("data/xsec.conf")
'''

xsections={}
for k in processes.items(): #processes is the list of processes and properties (files and, sometimes, weights)
                              #note: it's more or less a handmade list of folders that contain ntuple files, i.e.
                              #each QCD type, WW, WZ, CrabMC etc. it's the folders on the /2018 level
    #if v[1]=='MC': #if it's an MC file, then
       # xsections[k] = v[2] #make xsections[k] that is the file's weight
   # else:                   #otherwise,
    xsections[k] = -1 #make it -1

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

"""
this next chunk is key - it's what we need in order to actually create the json file of files
"""

os.system("mkdir -p beans") #creates target directory for final json file to land in
with open("beans/"+options.year+".json", "w") as fout: #sets up formatting for file that goes in that directory
    json.dump(datadef, fout, indent=4) #outputs to json file

"""
transition to getNTupleFiles from Weinan's utils/commonHelpers
"""
'''
from folders import *

def getNTupleFiles(
    gen_batch=-1,
    year=[2018], #originally this was 2016 or 2018, but that limits us to only signal files
    prefix="/store/group/lpcmetx/MCSIDM/ffNtuple/",
    xrootd=True,
):

    subdirs = []
    for y in year:
        for f in folderNames:
            year_dir = os.path.join(prefix, str(y), folderNames[f])
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
'''