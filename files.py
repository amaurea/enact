"""This module provides low-level access to the actpol TOD metadata files."""
import ast, numpy as np
from enlib.utils import lines

def read_gain(fname):
	"""Reads per-detector gain values from file, returning [id,val]."""
	data = read_pylike_format(fname)
	return np.vstack([data["det_uid"],data["cal"]])

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
	(the negative ones). The format is returned as [id,val]."""
	res = []
	for line in lines(fname):
		if line.startswith("#"): continue
		toks = line.split()
		id, ang = int(toks[0]), float(toks[3])*np.pi/180
		if ang < 0: continue
		res.append([id,ang])
	return np.array(res).T

def read_tconst(fname):
	"""Reads time constants from file, discarding those marked bad.
	Returns format [id,val]"""
	res = np.loadtxt(fname).T
	return res[:2,res[2]==0]

def read_point(fname):
	"""Reads the per-detector pointing offsets, returning it in the form [id,dx,dy]."""
	res = np.loadtxt(fname, usecols=[0,1,2,4]).T
	return res[:,res[1]>0][[0,2,3]]

def read_point_offset(fname):
	"""Reads per-tod pointing offsets, returning it in the form {todID: [dx,dy])."""
	res = {}
	for line in lines(fname):
		if line[0] == '#': continue
		toks = line.split()
		res[toks[0]] = np.array([float(toks[5]),float(toks[6])])
	return res

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
