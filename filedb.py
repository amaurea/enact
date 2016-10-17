import numpy as np, re, shlex, pipes, os
from enlib import filedb, config, bunch
from enlib.utils import ctime2date
from enact import todinfo

def id2ts(id): return int(id[:id.index(".")])
season_ends = [1390000000, 1421000000, 1454000000, 1490000000]

extractors = {
	"id":     lambda id: id,
	"ar":     lambda id: id[-1],
	"season": lambda id: 1+np.searchsorted(season_ends, id2ts(id)),
	"syear":  lambda id: 2013+np.searchsorted(season_ends, id2ts(id)),
	"t5":     lambda id: id[:5],
	"t":      lambda id: id[:id.index(".")],
	"date":   lambda id: ctime2date(id2ts(id), -9),
	"year":   lambda id: ctime2date(id2ts(id), -9, "%Y"),
	"month":  lambda id: ctime2date(id2ts(id), -9, "%m"),
	"day":    lambda id: ctime2date(id2ts(id), -9, "%d"),
	"Udate":  lambda id: ctime2date(id2ts(id),  0),
	"Uyear":  lambda id: ctime2date(id2ts(id),  0, "%Y"),
	"Umonth": lambda id: ctime2date(id2ts(id),  0, "%m"),
	"Uday":   lambda id: ctime2date(id2ts(id),  0, "%d"),
}

# Try to set up default databases. This is optional, and the databases
# will be none if it fails.
config.default("root", ".", "Path to directory where the different metadata sets are")
config.default("dataset", ".", "Path to data set directory relative to data_root")
config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata. Relative to dataset path.")
config.default("todinfo", "todinfo.hdf","File describing location of the TOD id lists. Relative to dataset path.")
config.default("file_override", "none", "Comma-separated list of field:file, or none to disable")
config.init()

class ACTFiles(filedb.FormatDB):
	def __init__(self, file=None, data=None, override=None):
		if file is None and data is None: file = cjoin(["root","dataset","filedb"])
		override = config.get("file_override", override)
		filedb.FormatDB.__init__(self, file=file, data=data, funcs=extractors, override=override)

def cjoin(names): return os.path.join(*[config.get(n) for n in names])

def init():
	global scans, data
	scans = todinfo.read(cjoin(["root","dataset","todinfo"]))
	data  = ACTFiles()
