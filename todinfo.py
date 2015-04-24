"""This module builds lists of TOD ids based on a set of tagged files and
a set of selectors. These are specified in the form of a dict
{file: set(tags), file:set(tags), ...}. Each file is assumed to have the
format [id] [hour] [el] [az] [pwv] [status].

Based on the file specification, a simple databsase is built by expanding and
combining each file into a list of fields and tags.

The database is queried using strings of the type "tag,tag,tag,...:sort[slice]",
where tag can be
 1. an actual tag. This requires the ids selected to be in a file with that tag
 2. tag+tag+...: This requires the ids to be in files that contain at least one of those tags
 3. an expression involving the fields in the file, like pwv<1 or hour>11.
sort specifies a field to sort the list by.
slice is either a normal python slice, where the constant n specifies the length of the array,
or the form a/b, indicating the a'th block out of b equally sized blocks, counting from zero.
This is just syntactic sugar for [a*n/b:(a+1)*n/b], since this is such a common case.

Here are some examples in the context of actpol analysis
 1. deep6                    all files for the target deep6
 2. deep6,s13                all season 2013 files for deep6
 3. deep6,s14,ar2            all season 2014 files for deep6 with array 2
 4. deep6:t[0::4]            every fourth file of deep6 after sorting by time
 5. deep6,s14:t[1*n/4:2*n/4] the second quarter of the 2014 deep6 data by time
 6. deep6,s14:t[1/4]         shorter way of writing the above
 7. deep6,night              all night-time deep6 data. Night is an automatic tag based on the hour field
 8. deep6,el>50,pwv<1        all deep6 files with el > 50 degrees and pwv < 1 mm
 9. deep6,pwv<2:pwv[0/2]     the lowest half of the files with pwv < 2 mm for deep6"""
import shlex, numpy as np, hashlib
from enlib import utils
from bunch import Bunch

def id2hash(id):
	toks = id.split(".")
	return hashlib.md5(toks[0]).hexdigest() + "." + toks[-1]

class TODinfo:
	def __repr__(self):
		return "TODinfo(fields="+ str(self.fields) + ", tags=" + str(self.tags) + ")"

class TODDB:
	def __init__(self, filespec, restrict_status=True):
		"""TOD databse which allows you to easily get ids of tods fulfilling
		various criteria. Construct either by passing the file name to a todinfo
		file with lines of the format [filename] [tag] [tag] ... as parsed by
		parse_todinfofile, or by directly specifying a filespec dictionary
		{filename: tagset}. restrict_status indicates whether tods with status
		less than 2 should be excluded as invalid.

		Example usage:
			db = TODDB("todinfo.txt")
			ids = db["deep56,ar2,night:t[0/2]"].ids
			for id in ids:
				do something with id"""
		if isinstance(filespec, TODDB):
			self.fields = filespec.fields.copy()
			self.tags = filespec.tags.copy()
		else:
			if isinstance(filespec, basestring):
				filespec = parse_todinfofile(filespec)
			fieldnames = ["id","hour","el","az","pwv","status"]
			fieldtypes = [str,float,float,float,float,int]
			self.fields = Bunch({n:[] for n in fieldnames})
			self.tags = []
			for fname, ftags in filespec.items():
				ftags = set(ftags)
				with open(fname, "r") as f:
					for line in f:
						if not line or len(line) < 1 or line[0] == "#": continue
						toks = line.split()
						if restrict_status and int(toks[5]) < 2: continue
						for n,typ,v in zip(fieldnames,fieldtypes,toks):
							self.fields[n].append(typ(v))
						# Automatically computed tags
						dn = "night" if self.fields["hour"][-1] < 11 else "day"
						self.tags.append(ftags | set([dn,self.fields["id"][-1]]))
			for k in self.fields.keys():
				self.fields[k] = np.array(self.fields[k])
			# Extra fields
			self.fields["t"] = np.array([float(v[:v.index(".")]) for v in self.fields["id"]])
			self.fields["hash"] = np.array([id2hash(v) for v in self.fields["id"]])
			self.fields["mjd"] = utils.ctime2mjd(self.fields["t"])
			self.fields["jon"] = calc_jon_day(self.fields["t"])
			self.tags = np.array(self.tags)
			# Sort by t by default
			inds = np.argsort(self.fields["t"])
			self.tags = self.tags[inds]
			for k in self.fields.keys():
				self.fields[k] = self.fields[k][inds]
	@property
	def n(self): return len(self.tags)
	@property
	def ids(self): return self.fields["id"]
	def copy(self): return TODDB(self)
	def select_inds(self, inds):
		res = self.copy()
		for k in res.fields.keys():
			res.fields[k] = res.fields[k][inds]
		res.tags = res.tags[inds]
		return res
	def select_tags(self, tags):
		try: tags = set(tags)
		except TypeError: tags = set([tags])
		return self.select_inds([i for i,otags in enumerate(self.tags) if tags & otags])
	def query(self, q): return query_db(self, q)
	def __getitem__(self, q):
		if isinstance(q, (int,long)):
			res = TODinfo()
			res.fields = Bunch()
			for f in self.fields:
				res.fields[f] = self.fields[f][q]
			res.tags = self.tags[q]
			return res
		else:
			return self.query(q)
	def __str__(self): return self.__repr__(100)
	def __repr__(self, nmax=None):
		lines = []
		n1, n2 = (self.n, 0) if not nmax or self.n <= nmax else (nmax/4, nmax/4)
		def pline(i):
			line = "%s %5.2f %5.2f %5.2f %5.2f %d" % tuple([self.fields[k][i] for k in ["id","hour","el","az","pwv","status"]])
			return line + " " + " ".join(sorted(list(self.tags[i])))
		for i in range(0,n1):
			lines.append(pline(i))
		if n2 > 0:
			lines.append("       ...       ")
			for i in range(self.n-n2, self.n):
				lines.append(pline(i))
		return "\n".join(lines)
	def __iter__(self):
		for i in xrange(self.n):
			yield self[i]

def parse_todinfofile(fname):
	res  = {}
	vars = {}
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if not line or len(line) < 1 or line[0] == "#": continue
			toks = shlex.split(line)
			assert len(toks) > 1, "Tod info entry needs at least one tag: '%s'" % line
			if toks[1] == "=":
				vars[toks[0]] = toks[2]
			else:
				res[toks[0].format(**vars)] = set(toks[1:])
	return res

def query_db(db, query):
	if query is None: return db # Null query returns unmodified db
	toks = query.split(":")
	if len(toks) == 0: return db # Empty selection returns unmofified db
	taglist, rest = toks[0], ":".join(toks[1:])
	if taglist:
		for tagexpr in taglist.split(","):
			try:
				# Copy to avoid having __builtins__ being inserted into fields
				locs = np.__dict__.copy()
				locs["int"] = np.int0
				locs["float"] = np.float_
				db = db.select_inds(np.where(eval(tagexpr, db.fields.copy(), locs))[0])
			except (NameError, AttributeError):
				db = db.select_tags(tagexpr.split("+"))
	if rest:
		try:
			i = rest.index("[")
			sort_key, s = rest[:i], rest[i+1:-1]
		except ValueError:
			sort_key, s = rest, ""
		n = db.n
		inds = np.arange(n)
		if sort_key: inds = np.argsort(db.fields[sort_key])
		if s:
			# Check for simplified block slice syntax
			try:
				i = s.index("/")
				a, b = int(s[:i]),int(s[i+1:])
				inds = inds[a*n/b:(a+1)*n/b]
			except ValueError:
				# Fall back on full format
				inds = eval("inds["+s+"]")
		db = db.select_inds(inds)
	return db

def calc_jon_day(ctime):
	secs = np.sort(ctime%86400)
	if len(ctime == 0): return ctime.astype(bool)
	elif len(ctime == 1): return secs/3600>11
	gaps = secs[1:]-secs[:-1]
	i = np.argmax(gaps)
	if secs[0]+86400-secs[-1] > gaps[i]:
		cut = 0
	else:
		cut = 0.5*(secs[i]+secs[i+1])
	return (ctime-cut)/86400

def get_tods(selector, dbfile):
	try:
		return utils.read_lines(selector)
	except IOError:
		return TODDB(dbfile)[selector].ids
