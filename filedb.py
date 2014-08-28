import numpy as np, re, bunch, shlex, datetime, pipes
from enlib import filedb

def pat_fixed(id, args): return args[0]
def pat_slice(id, args): return "%s/%s/%s%s" % (args[0], eval("id"+args[1]), id, "".join(args[2:]))
def pat_date(id, args):
	timestamp = int(id[:id.index(".")])
	tzone     = int(args[1])
	date      = datetime.datetime.utcfromtimestamp(timestamp+tzone*3600).strftime("%Y-%m-%d")
	return "%s/%s/%s%s" % (args[0], date, id, "".join(args[2:]))

patterns = {
		"fixed": pat_fixed,
		"slice": pat_slice,
		"date":  pat_date,
	}

class ACTdb(filedb.Basedb):
	def load(self, data):
		self.rules = []
		for line in data.splitlines():
			if len(line) < 1 or line[0] == "#": continue
			toks = shlex.split(line)
			name, pattern, args = toks[0][:-1], toks[1], toks[2:]
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
