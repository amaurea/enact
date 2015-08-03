"""This module provides low-level access to the actpol TOD metadata files."""
import ast, numpy as np, enlib.rangelist, re
from bunch import Bunch
from enlib.utils import lines
from enlib.zgetdata import dirfile

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
		if len(toks) > 2:
			id, ang = int(toks[0]), float(toks[3])*np.pi/180
		else:
			id, ang = int(toks[0]), float(toks[1])*np.pi/180
		if ang < 0: continue
		ids.append(id)
		res.append(ang)
	return np.array(ids), np.array(res)

def read_tconst(fname):
	"""Reads time constants from file, discarding those marked bad.
	Returns format id,val"""
	res  = np.loadtxt(fname).T
	good = res[1]>0
	res  = res[:2,good]
	return res[0].astype(int), res[1]

def read_point_template(fname):
	"""Reads the per-detector pointing offsets, returning it in the form id,[[dx,dy]]."""
	res = np.loadtxt(fname, usecols=[0,1,2,4]).T
	res = res[:,res[1]>0][[0,2,3]]
	return res[0].astype(int), res[1:].T

def read_point_offsets(fname):
	"""Reads per-tod pointing offsets, returning it in the form {todID: [dx,dy])."""
	res = {}
	for line in lines(fname):
		if line[0] == '#': continue
		toks = line.split()
		res[toks[0]] = np.array([float(toks[5]),float(toks[6])])
	return res

def read_cut(fname):
	"""Reads the act cut format, returning ids,cuts,offset, where cuts is a Multirange
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
	return oids, enlib.rangelist.Multirange(ocuts), offset

def read_site(fname):
	"""Given a filename or file, parse a file with key = value information and return
	it as a Bunch."""
	res = Bunch()
	for line in lines(fname):
		if line.isspace(): continue
		a = ast.parse(line)
		id = a.body[0].targets[0].id
		res[id] = ast.literal_eval(a.body[0].value)
	return res

def read_tod(fname, ids=None, mapping=lambda x: [x/32,x%32], ndet=33*32):
	"""Given a filename or dirfile, reads the time ordered data from the file,
	returning ids,data. If the ids argument is specified, only those ids will
	be retrieved. The mapping argument defines the mapping between ids and
	actual fields in the file, and ndet specifies the maximum number of detectors.
	These can usually be ignored."""
	# Find which ids to read
	if ids is None: ids = np.arange(ndet)
	ids = np.asarray(ids)
	rowcol = ids if ids.ndim == 2 else np.asarray(mapping(ids))
	def read(dfile, rowcol):
		nsamp = dfile.spf("tesdatar%02dc%02d" % tuple(rowcol[:,0]))*dfile.nframes
		res   = np.empty([rowcol.shape[1],nsamp],dtype=np.int32)
		for i, (r,c) in enumerate(rowcol.T):
			# The four lowest bits are status flags
			res[i] = dfile.getdata("tesdatar%02dc%02d" % (r,c)) >> 4
			dfile.raw_close()
		return res
	if isinstance(fname, basestring):
		with dirfile(fname) as dfile:
			return ids, read(dfile, rowcol)
	else:
		return ids, read(fname, rowcol)

def read_tod_moby(fname, ids=None, mapping=lambda x: [x/32,x%32], ndet=33*32):
	import moby2
	if ids is None: ids = np.arange(ndet)
	tod = moby2.scripting.get_tod({'filename': fname, 'det_uid':ids})
	return ids, tod.data

def read_boresight(fname):
	"""Given a filename or dirfile, reads the timestamp, azimuth, elevation and
	encoder flags for the telescope's boresight. No deglitching or other corrections
	are performed. Returns [unix time,az (deg),el(deg)], flags."""
	def read(dfile):
		res = np.array([dfile.getdata("C_Time"),dfile.getdata("Enc_Az_Deg_Astro"),dfile.getdata("Enc_El_Deg")]), dfile.getdata("enc_flags")
		dfile.raw_close()
		return res
	if isinstance(fname, basestring):
		with dirfile(fname) as dfile:
			return read(dfile)
	else:
		return read(fname)

def read_boresight_moby(fname):
	import moby2
	tod = moby2.scripting.get_tod({'filename': fname, 'det_uid':[],'read_data':False})
	return np.array([tod.ctime, tod.az*180/np.pi, tod.alt*180/np.pi]), tod.enc_flags

def read_spikes(fname):
	"""Given a filename, reads the start, end and amplitude of the spikes described
	in the file. Spikes without start/end are ignored."""
	a = np.loadtxt(fname).T
	good = a[5] != 0
	return a[:,good][[4,5,2]]

def read_noise_cut(fname):
	"""Given a filename, reads the set of detectors to cut for each tod,
	returning it as a dictionary of id:detlist."""
	res = {}
	for line in lines(fname):
		if line[0] == '#': continue
		toks = line.split()
		res[toks[0]] = np.array([int(w) for w in toks[2:]],dtype=np.int32)
	return res

def read_pickup_cut(fname):
	"""Given a filename, reads cuts in the pickup cut format
	id scan_direction hex azpix1 azpix2 az1 az2 strength."""
	res = {}
	for line in lines(fname):
		if line[0] == '#': continue
		id, dir, hex, ap1, ap2, az1, az2, strength = line.split()
		if id not in res: res[id] = []
		res[id].append([int(dir),int(hex),float(az1),float(az2),float(strength)])
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
			if v == "'nan'": res[id][i] = np.nan
			elif v == "nan": res[id][i] = np.nan
	return res
