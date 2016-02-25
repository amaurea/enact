import numpy as np, re, shlex, datetime, pipes, os
from enlib import filedb, config, bunch
from enact.todinfo import TODDB

def id2ts(id): return int(id[:id.index(".")])
def ts2date(timestamp, tzone, fmt="%Y-%m-%d"):
	return datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime(fmt)

extractors = {
	"id":     lambda id: id,
	"ar":     lambda id: id[-1],
	"season": lambda id: 1 if id2ts(id) < 1390000000 else 2 if id2ts(id) < 1424200000 else 3,
	"t5":     lambda id: id[:5],
	"t":      lambda id: id[:id.index(".")],
	"date":   lambda id: ts2date(id2ts(id), -9),
	"year":   lambda id: ts2date(id2ts(id), -9, "%Y"),
	"month":  lambda id: ts2date(id2ts(id), -9, "%m"),
	"day":    lambda id: ts2date(id2ts(id), -9, "%d"),
}

class ACTFiles(filedb.FormatDB):
	def __init__(self, file=None, data=None):
		filedb.FormatDB.__init__(self, file=file, data=data, funcs=extractors)

# Try to set up default databases. This is optional, and the databases
# will be none if it fails.
config.default("root", ".", "Path to directory where the different metadata sets are")
config.default("dataset", ".", "Path to data set directory relative to data_root")
config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata. Relative to dataset path.")
config.default("todinfo", "todinfo.txt","File describing location of the TOD id lists. Relative to dataset path.")
config.init()

def cjoin(names): return os.path.join(*[config.get(n) for n in names])

def init():
	global scans, data
	scans = TODDB(cjoin(["root","dataset","todinfo"]))
	data  = ACTFiles(cjoin(["root","dataset","filedb"]))
