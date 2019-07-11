# Sample Management Tool

process.py contains the list of folders to look through, separated by process (i.e., DoubleMuon, CRABPrivateMC, QCD, etc.).

eventCounter.py contains methods to count the number of events a particular file iterated over
in order to calculate weight. From https://github.com/phylsix/FireHydrant/blob/promptdatalook/Notebooks/MC/Samples/generate.py

fileGrabber.py combs through given folders to find the most recent timestamped folders and output all root files into a single .json file,
along with the corresponding weights.

Background files have their proper weights calculated from cross-section, luminosity, and number of events.
Data and Signal files have both weights and cross-sections assigned as 1, but this can be easily changed.

Due to errors in certain files not being stored on the disk, QCD_Pt_30to50 has been masked out, but this can also be easily removed.

The program requires input of a pack size, or how many files to bundle together in the .json file, as well as a year to title the file. So, the execute command looks like:

`python fileGrabber.py --year 2018 --pack 75`
