"""This module provides low-level access to the actpol TOD metadata files."""
import ast, numpy as np, enlib.rangelist, re
from enlib.utils import lines

def read_gain(fname):
	"""Reads per-detector gain values from file, returning id,val."""
	data = read_pylike_format(fname)
	return np.array(data["det_uid"]), np.array(data["cal"])

def read_gain_correction(fname):
	"""Reads per-tod overall gain correction from file. Returns
	{todID: val}."""
	res = {}
	for line in lines(fname):
		if not line.startswith("#"):
			id, value = line.split()
			res[id] = float(value)
	return res

def read_polangle(fname):
	"""Reads polarization angles in radians, discarding ones marked bad
	(the negative ones). The format is returned as id,val."""
	ids, res = [], []
	for line in lines(fname):
		if line.startswith("#"): continue
		toks = line.split()
		id, ang = int(toks[0]), float(toks[3])*np.pi/180
		if ang < 0: continue
		ids.append(id)
		res.append(ang)
	return np.array(ids), np.array(res)

def read_tconst(fname):
	"""Reads time constants from file, discarding those marked bad.
	Returns format id,val"""
	res = np.loadtxt(fname).T
	res = res[:2,res[2]==0]
	return res[0].astype(int), res[1]

def read_point_template(fname):
	"""Reads the per-detector pointing offsets, returning it in the form id,[dx,dy]."""
	res = np.loadtxt(fname, usecols=[0,1,2,4]).T
	res = res[:,res[1]>0][[0,2,3]]
	return res[0].astype(int), res[1:]

def read_point_offsets(fname):
	"""Reads per-tod pointing offsets, returning it in the form {todID: [dx,dy])."""
	res = {}
	for line in lines(fname):
		if line[0] == '#': continue
		toks = line.split()
		res[toks[0]] = np.array([float(toks[5]),float(toks[6])])
	return res

def read_cut(fname):
	"""Reads the act cut format, returning ids,cuts, where cuts is a Multirange
	object."""
	ids, cuts = [], []
	header = re.compile(r"^(\w+) *= *(\w+)$")
	rowcol = re.compile(r"^(\d+) +(\d+)$")
	entry  = re.compile(r"^(?:.+ )?r(\d+)c(\d+):(.*)$")
	nsamp  = 0
	nmax   = 0
	offset = 0
	for line in lines(fname):
		m = rowcol.match(line)
		if m:
			nrow, ncol = int(m.group(1)), int(m.group(2))
			continue
		m = header.match(line)
		if m:
			key, val = m.groups()
			if key == "n_samp": nsamp = int(val)
			elif key == "samp_offset": offset = int(val)
			elif key == "n_row": nrow = int(val)
			elif key == "n_col": ncol = int(val)
			continue
		m = entry.match(line)
		if m:
			r, c, toks = int(m.group(1)), int(m.group(2)), m.group(3).split()
			id = r*ncol+c
			ranges = np.array([[int(i) for i in word[1:-1].split(",")] for word in toks])
			nmax   = max(nmax,max([sub[1] for sub in ranges]))
			ids.append(id)
			cuts.append(ranges)
			continue
	# If there is no cut information, assume *fully cut*. Also prune totally cut detectors
	if nsamp == 0: nsamp = nmax
	oids, ocuts = [], []
	for id, cut in zip(ids, cuts):
		if len(cut) > 1 or len(cut) == 1 and not (np.all(cut[0]==[0,0x7fffffff]) or np.all(cut[0]==[0,nsamp])):
			oids.append(id)
			ocuts.append(enlib.rangelist.Rangelist(cut,nsamp))
	return oids, enlib.rangelist.Multirange(ocuts)

def read_pylike_format(fname):
	"""Givnen a file with a simple python-like format with lines of foo = [num,num,num,...],
	return it as a dictionary of names->lists, while preserving nan values."""
	res = {}
	for line in lines(fname):
		if line.isspace(): continue
		a = ast.parse(line.replace("nan", "'nan'")) # Does not handle nan
		id = a.body[0].targets[0].id
		res[id] = ast.literal_eval(a.body[0].value)
		# reinsert all the nans. This assumes no nested lists
		for i, v in enumerate(res[id]):
			if v == "'nan'": res[id][i] = float(v)
	return res
