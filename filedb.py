import numpy as np, re, bunch, shlex, datetime, pipes
from enlib import filedb

def id2ts(id): return int(id[:id.index(".")])
def ts2date(timestamp, tzone):
	return datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime("%Y-%m-%d")

extractors = {
	"id":     lambda id: id,
	"ar":     lambda id: id[-1],
	"season": lambda id: 1 if id2ts(id) < 1390000000 else 2,
	"t5":     lambda id: id[:5],
	"date":   lambda id: ts2date(id2ts(id), -9),
}

class ACTFiles(filedb.FormatDB):
	def __init__(self, file=None, data=None):
		filedb.FormatDB.__init__(self, file=file, data=data, funcs=extractors)

# Old version below

def pat_fixed(id, args): return args[0]
def pat_flat(id, args):  return "%s/%s%s" % (args[0], id, "".join(args[1:]))
def pat_slice(id, args): return "%s/%s/%s%s" % (args[0], eval("id"+args[1]), id, "".join(args[2:]))
def pat_date(id, args):
	timestamp = int(id[:id.index(".")])
	tzone     = int(args[1])
	date      = datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime("%Y-%m-%d")
	return "%s/%s/%s%s" % (args[0], date, id, "".join(args[2:]))

patterns = {
		"fixed": pat_fixed,
		"flat":  pat_flat,
		"slice": pat_slice,
		"date":  pat_date,
	}

class ACTdb(filedb.Basedb):
	def load(self, data):
		self.rules = []
		for line in data.splitlines():
			if len(line) < 1 or line[0] == "#": continue
			toks = filedb.pre_split(line)
			name, pattern, args = toks[0], toks[1], toks[2:]
			fun = patterns[pattern]
			self.rules.append({"name":name, "pattern":pattern, "fun":fun, "args":args})
	def dump(self):
		lines = []
		for rule in self.rules:
			line = "%s: %s" % (rule["name"], rule["pattern"])
			for arg in rule["args"]:
				line += " " + pipes.quote(arg)
			lines.append(line)
		return "\n".join(lines)
	def __getitem__(self, id):
		res = bunch.Bunch()
		for rule in self.rules:
			res[rule["name"]] = rule["fun"](id, rule["args"])
		res.id = id
		return res
