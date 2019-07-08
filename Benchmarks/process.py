#!/usr/bin/env python

processes = {
	#data
	'DoubleMuon':('DoubleMuon','Data',1),

	#signal
	'CRAB_PrivateMC':('signal','MC',1),

	#TTJets
	'TTJets_TuneCP5_13TeV-madgraphMLM-pythia8':('TTJets','MC',2.87e00),

	#diboson
	'WW_TuneCP5_13TeV-pythia8':('WW','MC',5.75e-01),
	'WZ_TuneCP5_13TeV-pythia8':('WZ','MC',4.25e-01),
	'ZZ_TuneCP5_13TeV-pythia8':('ZZ','MC'3.68e-01),

	#triboson
	'WGG_5f_TuneCP5_13TeV-amcatnlo-pythia8':('WGG','MC',1.21e-01),
	'WWG_TuneCP5_13TeV-amcatnlo-pythia8':('WWG','MC',1.11e-02),
	'WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8':('WWW','MC',5.38e-02),
	'WWZ_TuneCP5_13TeV-amcatnlo-pythia8':('WWZ','MC',0.0),
	'WZG_TuneCP5_13TeV-amcatnlo-pythia8':('WZG','MC',1.32e-03),
	'WZZ_TuneCP5_13TeV-amcatnlo-pythia8':('WZZ','MC',1.37e-02),
	'ZZZ_TuneCP5_13TeV-amcatnlo-pythia8':('ZZZ','MC',3.53e-03),

	#DYJetsToLL
	'DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8':('DY_10_to_50','MC',2.41e01),
	'DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8':('DY_50','MC',0.0),

	#QCD
	'QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_15_to_20','MC'),
	'QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_20_to_30','MC'),
	'QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_30_to_50','MC'),
	'QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_50_to_80','MC'),
	'QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_80_to_120','MC'),
	'QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_20_to_170','MC'),
	'QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_170_to_300','MC'),
	'QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_300_to_470','MC'),
	'QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_470_to_600','MC'),
	'QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_600_to_800','MC'),
	'QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_800_to_1000','MC'),
	'QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8':('QCD_1000_to_inf','MC')

}