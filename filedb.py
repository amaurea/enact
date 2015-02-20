import numpy as np, re, bunch, shlex, datetime, pipes
from enlib import filedb, config
from enact.todinfo import TODDB

def id2ts(id): return int(id[:id.index(".")])
def ts2date(timestamp, tzone, fmt="%Y-%m-%d"):
	return datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime(fmt)

extractors = {
	"id":     lambda id: id,
	"ar":     lambda id: id[-1],
	"season": lambda id: 1 if id2ts(id) < 1390000000 else 2,
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
# will be none if it fails
config.default("filedb", "filedb.txt", "File describing the location of the TOD and their metadata")
config.default("todinfo", "todinfo.txt","File describing location of the TOD id lists")
config.init()
try: scans = TODDB(config.get("todinfo"))
except IOError: scans = None
try: data  = ACTFiles(config.get("filedb"))
except IOError: data  = None
