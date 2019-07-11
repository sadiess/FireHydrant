#!/usr/bin/env python

processes = {
	#data
	'DoubleMuon': ('Data', 1),

	#signal
	'CRAB_PrivateMC': ('MC', 1),

	#TTJets
	'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8': ('MC', 491),

	#diboson
	'WW_TuneCP5_13TeV-pythia8': ('MC', 75.91),
	'WZ_TuneCP5_13TeV-pythia8': ('MC', 27.55),
	'ZZ_TuneCP5_13TeV-pythia8': ('MC', 12.14),

	#triboson
	'WGG_5f_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 2.001),
	'WWG_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.2316),
	'WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.2154),
	'WWZ_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.1676),
	'WZG_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.04345),
	'WZZ_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.05701),
	'ZZZ_TuneCP5_13TeV-amcatnlo-pythia8': ('MC', 0.01473),

	#DYJetsToLL
	'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8': ('MC', 15820),
	'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8': ('MC', 5317),

	#QCD
	'QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 2805000),
	'QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 2536000),
	'QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 1375000),
	'QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 377900),
	'QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 89730),
	'QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 21410),
	'QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 7022),
	'QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 619.8),
	'QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 59.32),
	'QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 18.19),
	'QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 3.271),
	'QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8': ('MC', 1.08)

}