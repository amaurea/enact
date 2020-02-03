from __future__ import division, print_function
import numpy as np, re, shlex, pipes, os
from enlib import filedb, config, bunch, execdb
from enlib.utils import ctime2date
from enact import todinfo

def id2ts(id): return int(id[:id.index(".")])
season_ends = [1390000000, 1421000000, 1454000000, 1490000000, 1520000000]

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
config.default("filevars", "filevars.py", "File with common definitions for filedbs")
config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata. Relative to dataset path.")
config.default("todinfo", "todinfo.hdf","File describing location of the TOD id lists. Relative to dataset path.")
config.default("file_override", "none", "Comma-separated list of field:file, or none to disable")
config.default("patch_dir", "area", "Directory where standard patch geometries are stored.")
config.init()

#class ACTFiles(filedb.FormatDB):
#	def __init__(self, file=None, data=None, override=None):
#		if file is None and data is None: file = cjoin(["root","dataset","filedb"])
#		override = config.get("file_override", override)
#		filedb.FormatDB.__init__(self, file=file, data=data, funcs=extractors, override=override)

def setup_filedb():
	"""Create a default filedb based on the root, dataset and filedb config
	variables. The result will be either a FormatDB or ExecDB based on the
	format of the fildb file."""
	override= config.get("file_override")
	if override is "none": override = None
	return execdb.ExecDB(cjoin(["root","dataset","filedb"]), cjoin(["root","filevars"]), override=override, root=cjoin(["root"]))

def cjoin(names): return os.path.join(*[config.get(n) for n in names])

def get_path_path(name):
	return cjoin(["root","patch_dir",name, ".fits"])

def init():
	global scans, data
	scans = todinfo.read(cjoin(["root","dataset","todinfo"]), vars={"root":cjoin(["root"])})
	data  = setup_filedb()
