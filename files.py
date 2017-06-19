"""This module provides low-level access to the actpol TOD metadata files."""
import ast, numpy as np, enlib.rangelist, re, multiprocessing, h5py
from enlib import pyactgetdata, zgetdata, bunch, utils, flagrange

def read_gain(fname):
	"""Reads per-detector gain values from file, returning id,val."""
	data = read_pylike_format(fname)
	return np.array(data["det_uid"]), np.array(data["cal"])

def read_gain_correction(fname, id=None):
	"""Read lines of the format id[:tag] val or id tag val. Returns it as a dict
	of {id: {tag:val,...}}. So a single TOD may be covered by multiple
	entries in the file, each of which covers a different subset.
	Lines that start with # will be ignored. If the id argument is
	passed in, only lines with matching id will be returned."""
	res = {}
	for line in utils.lines(fname):
		if line.startswith("#"): continue
		if id and not line.startswith(id) and not line.startswith("*"): continue
		# Parse the line
		line = line.replace(":"," ")
		toks = line.split()
		if len(toks) == 2:
			tod_id, value = toks
			tag = "*"
		else:
			tod_id, tag, value = toks
		value = float(value)
		# And insert it at the right location
		if tod_id not in res: res[tod_id] = {}
		res[tod_id][tag] = value
	return res

#def read_gain_correction(fname, id=None):
#	"""Reads per-tod overall gain correction from file. Returns
#	{todID: val}."""
#	res = {}
#	for line in utils.lines(fname):
#		if line.startswith("#"): continue
#		if id and not line.startswith(id) and not line.startswith("*"): continue
#		tod_id, value = line.split()
#		res[tod_id] = float(value)
#	return res

def read_polangle(fname, mode="auto"):
	"""Reads polarization angles in radians, discarding ones marked bad
	(the negative ones). The format is returned as id,val."""
	ids, res = [], []
	for line in utils.lines(fname):
		if line.startswith("#"): continue
		toks = line.split()
		if mode == "irca" or len(toks) > 2:
			id, ang = int(toks[0]), float(toks[3])*np.pi/180
		else:
			id, ang = int(toks[0]), float(toks[1])*np.pi/180
		if ang < 0: continue
		ids.append(id)
		res.append(ang)
	return np.array(ids), np.array(res)

#def read_tconst(fname):
#	"""Reads time constants from file, discarding those marked bad.
#	Returns format id,val"""
#	res  = np.loadtxt(fname).T
#	good = res[1]>0
#	res  = res[:2,good]
#	return res[0].astype(int), res[1]

def read_tconst(fname, id=None, mode="auto"):
	if mode == "hdf" or mode == "auto" and fname.endswith(".hdf"):
		return read_tconst_hdf(fname, id=id)
	else:
		return read_tconst_ascii(fname, mode=mode)

def read_tconst_ascii(fname, mode="auto"):
	"""Reads time constants from file in one of two formats:
	[uid,tau,_] and [uid,row,col,f3db,std]. Values of 0 are taken
	to indicate a bad value, and are discarded. Returns dets, taus,
	regardless of whether the input was taus or f3dbs."""
	dets, taus = [], []
	with open(fname, "r") as f:
		for line in f:
			if line.startswith('#'): continue
			toks = line.split()
			if len(toks) == 0: continue
			if mode == "tau" or mode == "auto" and (len(toks) == 2 or len(toks) == 3):
				det, tau = int(toks[0]), float(toks[1])
				if tau > 0:
					dets.append(det)
					taus.append(tau)
			elif len(toks) == 5:
				det, f3db = int(toks[0]), float(toks[3])
				if f3db > 0:
					tau = 1/(2*np.pi*f3db)
					dets.append(det)
					taus.append(tau)
			else:
				raise IOError
	return np.array(dets), np.array(taus)

def read_tconst_hdf(fname, id):
	with h5py.File(fname, "r") as hfile:
		ids = hfile["id"].value
		ind = np.where(ids == id)[0]
		if len(ind) == 0: raise IOError
		ind = ind[0]
		taus = hfile["tau"][ind]
		dets = np.where(taus>0)[0]
		taus = taus[dets]
		return dets, taus

def read_point_template(fname):
	"""Reads the per-detector pointing offsets, returning it in the form id,[[dx,dy]]."""
	res = np.loadtxt(fname, usecols=[0,1,2,4]).T
	res = res[:,res[1]>0][[0,2,3]]
	return res[0].astype(int), res[1:].T

def read_point_offsets(fname):
	"""Reads per-tod pointing offsets, returning it in the form {todID: [dx,dy])."""
	res = {}
	for line in utils.lines(fname):
		if line[0] == '#': continue
		toks = line.split()
		id   = ".".join(toks[0].split(".")[:3])
		res[id] = np.array([float(toks[5]),float(toks[6])])
	return res

def read_cut(fname):
	"""Read the act cut format, returning ids, cuts, offset, where cuts is a Multirange
	object."""
	nsamp, ndet, offset = None, None, None
	dets, cuts = [], []
	for line in utils.lines(fname):
		if "=" in line:
			# Header key-value pair
			toks = line.split()
			if   toks[0] == "n_det":  ndet  = int(toks[2])
			elif toks[0] == "n_samp": nsamp = int(toks[2])
			elif toks[0] == "samp_offset": offset = int(toks[2])
			else: continue # Ignore others
		elif ":" in line:
			parts = line.split(":")
			uid   = int(parts[0].split()[0])
			if len(parts) > 1 and "(" in parts[1]:
				toks  = parts[1].split()
				ranges = np.array([[int(w) for w in tok[1:-1].split(",")] for tok in toks])
				ranges = np.minimum(ranges, nsamp)
			# Handle uncut detectors
			else:
				ranges = np.zeros([0,2],dtype=int)
			dets.append(uid)
			cuts.append(enlib.rangelist.Rangelist(ranges,nsamp))
	# Filter out fully cut tods
	odets, ocuts = [], []
	for det, cut in zip(dets, cuts):
		if cut.sum() < cut.n:
			odets.append(det)
			ocuts.append(cut)
	ocuts = enlib.rangelist.Multirange(ocuts)
	return odets, ocuts, offset

def read_cut_hdf(fname, id, flags):
	"""Reads cuts in the new hdf format. This format has multiple tods per
	file, so the tod id must be specified. It also lets one specify various
	flags to construct the actual cuts from, such as planet cuts, glitch cuts, etc."""
	frange = flagrange.read_flagrange(fname, id)
	frange = frange.select(flags)
	cuts   = frange.to_rangelist()
	return frange.dets, cuts, frange.sample_offset

#def read_cut(fname):
#	"""Reads the act cut format, returning ids,cuts,offset, where cuts is a Multirange
#	object."""
#	ids, cuts = [], []
#	header = re.compile(r"^(\w+) *= *(\w+)$")
#	rowcol = re.compile(r"^(\d+) +(\d+)$")
#	entry  = re.compile(r"^(?:.+ )?r(\d+)c(\d+):(.*)$")
#	nsamp  = 0
#	nmax   = 0
#	offset = 0
#	for line in utils.lines(fname):
#		m = rowcol.match(line)
#		if m:
#			nrow, ncol = int(m.group(1)), int(m.group(2))
#			continue
#		m = header.match(line)
#		if m:
#			key, val = m.groups()
#			if key == "n_samp": nsamp = int(val)
#			elif key == "samp_offset": offset = int(val)
#			elif key == "n_row": nrow = int(val)
#			elif key == "n_col": ncol = int(val)
#			continue
#		m = entry.match(line)
#		if m:
#			r, c, toks = int(m.group(1)), int(m.group(2)), m.group(3).split()
#			id = r*ncol+c
#			ranges = np.array([[int(i) for i in word[1:-1].split(",")] for word in toks])
#			nmax   = max(nmax,np.max(ranges[:,1]))
#			# Cap to nsamp if available
#			if nsamp: ranges[:,1] = np.minimum(nsamp, ranges[:,1])
#			ranges[:,0] = np.maximum(0, ranges[:,0])
#			ids.append(id)
#			cuts.append(ranges)
#			continue
#	# If there is no cut information, assume *fully cut*. Also prune totally cut detectors
#	if nsamp == 0: nsamp = nmax
#	oids, ocuts = [], []
#	for id, cut in zip(ids, cuts):
#		if len(cut) > 1 or len(cut) == 1 and not (np.all(cut[0]==[0,0x7fffffff]) or np.all(cut[0]==[0,nsamp])):
#			oids.append(id)
#			ocuts.append(enlib.rangelist.Rangelist(cut,nsamp))
#	return oids, enlib.rangelist.Multirange(ocuts), offset

def write_cut(fname, dets, cuts, offset=0, nrow=33, ncol=32):
	ndet, nsamp = cuts.shape
	ntot = nrow*ncol
	lines = [
		"format = 'TODCuts'",
		"format_version = 1",
		"n_det = %d" % ntot,
		"n_row = %d" % nrow,
		"n_col = %d" % ncol,
		"n_samp = %d" % nsamp,
		"samp_offset = %d" % offset,
		"END"]
	detinds = np.zeros(ntot,dtype=int)
	detinds[dets] = np.arange(len(dets))+1
	for uid, di in enumerate(detinds):
		row, col = uid/ncol, uid%ncol
		if di == 0:
			ranges = [[0,nsamp]]
		else:
			ranges = cuts[di-1].ranges
		lines.append("%4d r%02dc%02d: " % (uid, row, col) + " ".join(["(%d,%d)" % tuple(r) for r in ranges]))
	with open(fname, "w") as f:
		f.write("\n".join(lines))

def read_site(fname):
	"""Given a filename or file, parse a file with key = value information and return
	it as a Bunch."""
	res = bunch.Bunch()
	for line in utils.lines(fname):
		if line.isspace(): continue
		a = ast.parse(line)
		id = a.body[0].targets[0].id
		res[id] = ast.literal_eval(a.body[0].value)
	return res

def read_layout(fname):
	"""Read the detector layout, returning a Bunch of with
	ndet, nrow, ncol, rows, cols, darksquid, pcb."""
	rows, cols, dark, pcb = [], [], [], []
	with open(fname,"r") as f:
		for line in f:
			if line.startswith("#"): continue
			toks = line.split()
			r, c, d, p = int(toks[1]), int(toks[2]), int(toks[3])>0, toks[4]
			rows.append(r)
			cols.append(c)
			dark.append(d)
			pcb.append(p)
	rows = np.array(rows)
	cols = np.array(cols)
	dark = np.array(dark)
	pcb  = np.array(pcb)
	return bunch.Bunch(rows=rows, cols=cols, dark=dark, pcb=pcb, nrow=np.max(rows)+1, ncol=np.max(cols)+1, ndet=len(rows))

def read_tod(fname, ids=None, mapping=lambda x: [x/32,x%32], ndet=None, shape_only=False, nthread=1):
	"""Given a filename or dirfile, reads the time ordered data from the file,
	returning ids,data. If the ids argument is specified, only those ids will
	be retrieved. The mapping argument defines the mapping between ids and
	actual fields in the file, and ndet specifies the maximum number of detectors.
	These can usually be ignored. If nthread > 1, the tod fields will be read in parallel,
	which can give a significant speedup. If called this way, the function is not thread
	safe."""
	# Find which ids to read
	def get_ids(dfile, ids, ndet, mapping):
		if ids is None:
			fields = set(dfile.fields)
			id, ids = 0, []
			while "tesdatar%02dc%02d" % tuple(mapping(id)) in fields:
				if ndet is not None and id >= ndet: break
				ids.append(id)
				id += 1
		ids = np.asarray(ids)
		return ids
	def read(dfile, rowcol):
		global read_tod_single_dfile
		reference = rowcol[:,0] if rowcol.size > 0 else [0,0]
		nsamp = len(dfile.getdata("tesdatar%02dc%02d" % tuple(reference)))
		if shape_only: return nsamp
		res   = np.empty([rowcol.shape[1],nsamp],dtype=np.int32)
		if nthread == 1:
			for i, (r,c) in enumerate(rowcol.T):
				# The 7 lowest bits are status flags
				res[i] = dfile.getdata("tesdatar%02dc%02d" % (r,c))
		else:
			# Read in parallel, since there is a significant CPU cost to reading.
			# However, only do that when we use more than 1 proc, since this
			# parallelization has some over head (about 10%).
			def collect(args): res[args[0]] = args[1]
			read_tod_single_dfile = dfile
			pool = multiprocessing.Pool(nthread)
			for i, (r,c) in enumerate(rowcol.T):
				pool.apply_async(read_tod_single_helper, args=(i,r,c), callback=collect)
			pool.close()
			pool.join()
		return res
	if isinstance(fname, basestring):
		with pyactgetdata.dirfile(fname) as dfile:
			ids = get_ids(dfile, ids, ndet, mapping)
			rowcol = np.asarray(mapping(ids))
			return ids, read(dfile, rowcol)
	else:
		dfile = fname
		ids = get_ids(dfile, ids, ndet, mapping)
		rowcol = np.asarray(mapping(ids))
		return ids, read(dfile, rowcol)

# Helpers for parallel tod read
read_tod_single_dfile = None
def read_tod_single_helper(i, r, c):
	global read_tod_single_dfile
	# The 7 lowest bits are status flags
	return (i, read_tod_single_dfile.getdata("tesdatar%02dc%02d" % (r,c)))

def read_tod_moby(fname, ids=None, mapping=lambda x: [x/32,x%32], ndet=33*32, shape_only=False):
	import moby2
	if ids is None: ids = np.arange(ndet)
	if shape_only:
		foo = moby2.scripting.get_tod({'filename': fname, 'det_uid':ids[:1]})
		return ids, foo.data.size
	tod = moby2.scripting.get_tod({'filename': fname, 'det_uid':ids})
	return ids, tod.data

def read_boresight(fname):
	"""Given a filename or dirfile, reads the timestamp, azimuth, elevation and
	encoder flags for the telescope's boresight. No deglitching or other corrections
	are performed. Returns [unix time,az (deg),el(deg)], flags."""
	def read(dfile):
		res = np.array([dfile.getdata("C_Time"),dfile.getdata("Enc_Az_Deg_Astro"),dfile.getdata("Enc_El_Deg")]), dfile.getdata("enc_flags")
		return res
	if isinstance(fname, basestring):
		with pyactgetdata.dirfile(fname) as dfile:
			return read(dfile)
	else:
		return read(fname)

def read_boresight_moby(fname):
	import moby2
	tod = moby2.scripting.get_tod({'filename': fname, 'det_uid':[],'read_data':False})
	return np.array([tod.ctime, tod.az*180/np.pi, tod.alt*180/np.pi]), tod.enc_flags

def read_hwp_raw(fname):
	"""Given a filename or a dirfile, reads the half-wave-plate angle. May
	move this into read_boresight later, as it belongs with the other fields
	there."""
	def read(dfile):
		toks = dfile.fname.split(".")
		array = toks.pop()
		if array == "zip":
			array = toks.pop()
		field  = "hwp_pa%s_ang" % array[-1]
		if field in dfile.fields:
			return dfile.getdata(field)
		else:
			_, nsamp = read_tod(dfile, shape_only=True)
			return np.zeros(nsamp,dtype=np.int16)
	if isinstance(fname, basestring):
		with pyactgetdata.dirfile(fname) as dfile:
			return read(dfile)
	else:
		return read(fname)

def read_hwp_status(fname):
	res = {}
	with open(fname, "r") as f:
		for line in f:
			if line.startswith("#"): continue
			id, status = line.split()[:2]
			res[id] = int(status)
	return res

def read_hwp_epochs(fname):
	res = {}
	amap = {"PA1": "ar1", "PA2": "ar2", "PA3": "ar3"}
	with open(fname, "r") as f:
		for line in f:
			pa, name, t1, t2 = line.split()[:4]
			ar = amap[pa]
			if ar not in res: res[ar] = []
			res[ar].append([float(t1),float(t2),name])
	return res

def read_hwp_cleaned(fname, mode="auto"):
	"""Given a filename to an uncompressed dirfile containing hwp_angle_fit
	data as produced by Marius, return the hwp samples in degrees."""
	# Try Marius format
	if mode == "marius" or mode == "auto":
		try:
			with zgetdata.dirfile(fname) as dfile:
				nsamp = dfile.eof('hwp_angle_fit')
				return dfile.getdata("hwp_angle_fit", zgetdata.FLOAT32, num_samples=nsamp)
		except zgetdata.BadCodeError:
			if mode == "marius": raise errors.DataMissing("File %s is not in marius format" % fname)
	# Try the other format
	with pyactgetdata.dirfile(fname) as dfile:
		return dfile.getdata("Hwp_Angle")

def read_spikes(fname):
	"""Given a filename, reads the start, end and amplitude of the spikes described
	in the file. Spikes without start/end are ignored."""
	a = np.loadtxt(fname, ndmin=2).T
	good = a[5] != 0
	return a[:,good][[4,5,2]]

def read_noise_cut(fname, id=None):
	"""Given a filename, reads the set of detectors to cut for each tod,
	returning it as a dictionary of id:detlist."""
	res = {}
	for line in utils.lines(fname):
		if line[0] == '#': continue
		if id and not line.startswith(id): continue
		toks = line.split()
		res[toks[0]] = np.array([int(w) for w in toks[2:]],dtype=np.int32)
	return res

def read_pickup_cut(fname):
	"""Given a filename, reads cuts in the pickup cut format
	id scan_direction hex azpix1 azpix2 az1 az2 strength."""
	res = {}
	for line in utils.lines(fname):
		if line[0] == '#': continue
		id, dir, hex, ap1, ap2, az1, az2, strength = line.split()
		if id not in res: res[id] = []
		res[id].append([int(dir),int(hex),float(az1),float(az2),float(strength)])
	return res

def read_beam(fname):
	"""Given a filename, read an equi-spaced radial beam profile.
	The file should have format [r,b(r)]. [r,b(r)]"""
	return np.loadtxt(fname, ndmin=2).T

def read_dark_dets(fname):
	"""Read a list of detectors from a file with one uid per line. Returns
	a 1d numpy array of ints."""
	return np.loadtxt(fname).astype(int).reshape(-1)

def read_buddies(fname, mode="auto"):
	"""Read a beam decomposition of the near-sidelobe "buddies".
	Each line should contain xi eta T Q U for one buddy, or
	det xi eta T Q U for the detector-dependent format. The result
	will be dets, [ndet][nbuddy,{xi,eta,T,Q,U}]. For the
	buddy-independent format, dets is None."""
	res = np.loadtxt(fname, ndmin=2)
	if res.size == 0: return None, res.reshape(-1,5)
	if mode == "uniform" or mode == "auto" and res.shape[-1] == 5:
		# detector-independent format
		return None, [res]
	else:
		# detector-dependent format
		groups = utils.find_equal_groups(res[:,0])
		dets   = [int(res[g[0],0]) for g in groups]
		buds   = np.array([res[g,1:] for g in groups])
		return dets, buds

def read_apex(fname):
	"""Read weather data from apex from a gzip-compressed text file with
	columns [ctime] [value]."""
	return np.loadtxt(fname).reshape(-1,2)

def read_tags(fname):
	"""Read a set of detector tag definitions from file. Returns a
	dict[tag] -> array of ids."""
	res = {}
	for line in utils.lines(fname):
		toks = line.split()
		name = toks[0]
		ids  = np.array([int(tok) for tok in toks[1:]],dtype=int)
		res[name] = ids
	return res

def read_pylike_format(fname):
	"""Givnen a file with a simple python-like format with lines of foo = [num,num,num,...],
	return it as a dictionary of names->lists, while preserving nan values."""
	res = {}
	for line in utils.lines(fname):
		if line.isspace(): continue
		try:
			a = ast.parse(line.replace("nan", "'nan'")) # Does not handle nan
		except TypeError as e:
			raise IOError("Unparsable file %s (%s)" % (str(fname), e.message))
		id = a.body[0].targets[0].id
		res[id] = ast.literal_eval(a.body[0].value)
		# reinsert all the nans. This assumes no nested lists
		for i, v in enumerate(res[id]):
			if v == "'nan'": res[id][i] = np.nan
			elif v == "nan": res[id][i] = np.nan
	return res
