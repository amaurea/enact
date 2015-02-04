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
import copy.copy as copyfun, shlex

class TODDB:
	def __init__(self, filespec, restrict_status=True):
		if isinstance(filespec, basestring):
			filespec = parse_todinfofile(filespec)
		fieldnames = ["id","hour","el","az","pwv","status"]
		self.fields = {n:[] for n in fieldnames}
		self.tags = []
		for fname, ftags in filespec.items():
			ftags = set(ftags)
			with open(fname, "r") as f:
				for line in f:
					if not line or len(line) < 1 or line[0] == "#": continue
					toks = line.split()
					if restrict_status and toks[5] < 2: continue
					for n,v in zip(fieldnames,toks):
						self.fields[n].append(v)
					# Automatically computed tags
					dn = "night" if self.fields["hour"][-1] < 11 else "day"
					self.tags.append(ftags + set([dn]))
		for k in self.fields:
			self.fields[k] = np.array(self.fields)
		self.tags = np.array(self.tags)
	@property
	def n(self): return len(self.tags)
	@property
	def ids(self): return self.fields["id"]
	def copy(self): return copyfun(self)
	def select_inds(self, inds):
		res = self.copy()
		for k in res.fields:
			res.fields[k] = res.fields[k][inds]
		res.tags = res.tags[inds]
		return res
	def select_tag(self, tag):
		return select_inds([i for i,tags in enumerate(self.tags) if tag in tags])
	def query(self, q): return query_db(self, q)

def parse_todinfofile(fname):
	res = {}
	with open(fname,"r") as f:
		for line in f:
			if not line or len(line) < 1 or line[0] == "#": continue
			toks = shlex.split(line)
			res[toks[0]] = set(toks[1:])
	return res

def query_db(db, query):
	toks = query.split(":")
	taglist, rest = toks[0], ":".join(toks[1:])
	for tagexpr in taglist:
		try:
			db = db.select_inds(np.where(eval(tagexpr, db.fields))[0])
		except NameError:
			for tag in tagexpr.split("+"):
				db = db.select_tag(tag)
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
				inds = inds[i*n/b:(i+1)*n/b]
			except ValueError:
				# Fall back on full format
				inds = eval("inds"+s)
		db = db.select_inds(inds)
	return db
