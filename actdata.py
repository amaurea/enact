import numpy as np, time
from enlib import utils, dataset, nmat, config, errors, gapfill, fft, rangelist, zgetdata, pointsrcs
from enact import files, cuts, filters

def try_read(method, desc, fnames, *args, **kwargs):
	"""Try to read multiple alternative filenames, raising a DataMissing
	exception only if none of them can be read. Otherwise, return the first
	matching."""
	if isinstance(fnames, basestring): fnames = [fnames]
	for fname in fnames:
		try: return method(fname, *args, **kwargs)
		except (IOError,zgetdata.OpenError) as e: pass
	raise errors.DataMissing(desc + ": " + ", ".join(fnames))

def read_gain(entry):
	dets, gain_raw = try_read(files.read_gain, "gain", entry.gain)
	try: correction = try_read(files.read_gain_correction, "gain_correction", entry.gain_correction, id=entry.id)[entry.id]
	except KeyError: raise errors.DataMissing("gain_correction id: " + entry.id)
	mask = np.isfinite(gain_raw)*(gain_raw != 0)
	dets, gain_raw = dets[mask], gain_raw[mask]
	return dataset.DataSet([
		dataset.DataField("gain", gain_raw*correction, dets=dets, det_index=0),
		dataset.DataField("gain_raw", gain_raw, dets=dets, det_index=0),
		dataset.DataField("gain_correction", correction),
		dataset.DataField("entry", entry)])

def read_polangle(entry):
	dets, data = try_read(files.read_polangle, "polangle", entry.polangle)
	return dataset.DataSet([
		dataset.DataField("polangle", data, dets=dets, det_index=0),
		dataset.DataField("entry", entry)])

def read_tconst(entry):
	dets, data = try_read(files.read_tconst, "tconst", entry.tconst)
	return dataset.DataSet([
		dataset.DataField("tau", data, dets=dets, det_index=0),
		dataset.DataField("entry", entry)])

def read_cut(entry):
	dets, data, offset = try_read(files.read_cut, "cut", entry.cut)
	samples = [offset, offset + data.shape[-1]]
	return dataset.DataSet([
		dataset.DataField("cut", data, dets=dets, det_index=0, samples=samples, sample_index=1, stacker=rangelist.stack_ranges),
		dataset.DataField("entry", entry)])

def read_point_offsets(entry):
	dets, template = try_read(files.read_point_template, "point_template", entry.point_template)
	try: correction = try_read(files.read_point_offsets, "point_offsets", entry.point_offsets)[entry.id]
	except KeyError: raise errors.DataMissing("point_offsets id: " + entry.id)
	return dataset.DataSet([
		dataset.DataField("point_offset",  template+correction, dets=dets, det_index=0),
		dataset.DataField("point_template",template, dets=dets, det_index=0),
		dataset.DataField("point_correction",correction),
		dataset.DataField("entry", entry)])

def read_site(entry):
	site = try_read(files.read_site, "site", entry.site)
	return dataset.DataSet([
		dataset.DataField("site", site),
		dataset.DataField("entry", entry)])

def read_noise(entry):
	data = try_read(nmat.read_nmat, "noise", entry.noise)
	return dataset.DataSet([
		dataset.DataField("noise", data, dets=data.dets, det_index=0),
		dataset.DataField("entry", entry)])

def read_beam(entry):
	beam = try_read(files.read_beam, "beam", entry.beam)
	return dataset.DataSet([
		dataset.DataField("beam", beam),
		dataset.DataField("entry", entry)])

def read_noise_cut(entry):
	try: dets = try_read(files.read_noise_cut, "noise_cut", entry.noise_cut, id=entry.id)[entry.id]
	except KeyError: raise errors.DataMissing("noise_cut id: " + entry.id)
	return dataset.DataSet([
		dataset.DataField("noise_cut", dets=dets),
		dataset.DataField("entry", entry)])

def read_spikes(entry):
	spikes = try_read(files.read_spikes, "spikes", entry.spikes)
	return dataset.DataSet([
		dataset.DataField("spikes", data=spikes),
		dataset.DataField("entry", entry)])

def read_boresight(entry, moby=False):
	if moby: bore, flags = try_read(files.read_boresight_moby, "boresight", entry.tod)
	else:    bore, flags = try_read(files.read_boresight,      "boresight", entry.tod)
	return dataset.DataSet([
		dataset.DataField("boresight", bore, samples=[0,bore.shape[1]], sample_index=1),
		dataset.DataField("flags",     flags,samples=[0,flags.shape[0]],sample_index=0),
		dataset.DataField("entry",     entry)])

#def read_hwp(entry):
#	hwp = try_read(files.read_hwp, "hwp", entry.tod)
#	return dataset.DataSet([
#		dataset.DataField("hwp", hwp, samples=[0,hwp.size], sample_index=0)])

def read_hwp(entry):
	dummy = dataset.DataSet([
		dataset.DataField("hwp", 0),
		dataset.DataField("hwp_id", "none")])
	epochs = try_read(files.read_hwp_epochs, "hwp_epochs", entry.hwp_epochs)
	t, _, ar = entry.id.split(".")
	t = float(t)
	if ar not in epochs: return dummy
	for epoch in epochs[ar]:
		if t >= epoch[0] and t < epoch[1]:
			# Ok, the HWP was active during this period. So our data is missing
			# if we can't read it.
			status = try_read(files.read_hwp_status, "hwp_status", entry.hwp_status)
			if entry.id not in status or status[entry.id] != 1:
				raise errors.DataMissing("Missing HWP angles!")
			# Try to read the angles themselves
			hwp = try_read(files.read_hwp_cleaned, "hwp_angles", entry.hwp)
			return dataset.DataSet([
				dataset.DataField("hwp", hwp, samples=[0,hwp.size], sample_index=0),
				dataset.DataField("hwp_id", epoch[2])])
	# Not in any epoch, so return 0 hwp angle (which effectively turns it off)
	return dummy

def read_layout(entry):
	data = try_read(files.read_layout, "layout", entry.layout)
	return dataset.DataSet([
		dataset.DataField("layout", data),
		dataset.DataField("entry", entry)])

def read_pointsrcs(entry):
	data = try_read(pointsrcs.read, "pointsrcs", entry.pointsrcs, exact=False)
	return dataset.DataSet([
		dataset.DataField("pointsrcs", data),
		dataset.DataField("entry", entry)])

def read_tod_shape(entry, moby=False):
	if moby: dets, nsamp = try_read(files.read_tod_moby, "tod_shape", entry.tod, shape_only=True)
	else:    dets, nsamp = try_read(files.read_tod,      "tod_shape", entry.tod, shape_only=True)
	return dataset.DataSet([
		dataset.DataField("tod_shape", dets=dets, samples=[0,nsamp]),
		dataset.DataField("entry", entry)])

def read_tod(entry, dets=None, moby=False):
	if moby: dets, tod = try_read(files.read_tod_moby, "tod", entry.tod, ids=dets)
	else:    dets, tod = try_read(files.read_tod,      "tod", entry.tod, ids=dets)
	return dataset.DataSet([
		dataset.DataField("tod", tod, dets=dets, samples=[0,tod.shape[1]], det_index=0, sample_index=1, force_contiguous=True),
		dataset.DataField("entry", entry)])

readers = {
		"gain": read_gain,
		"polangle": read_polangle,
		"tconst": read_tconst,
		"cut": read_cut,
		"point_offsets": read_point_offsets,
		"pointsrcs": read_pointsrcs,
		"layout": read_layout,
		"beam": read_beam,
		"site": read_site,
		"noise": read_noise,
		"noise_cut": read_noise_cut,
		"spikes": read_spikes,
		"boresight": read_boresight,
		"tod_shape": read_tod_shape,
		"tod": read_tod,
		"hwp": read_hwp,
	}

default_fields = ["layout","beam","gain","polangle","tconst","cut","point_offsets","site","spikes","boresight","hwp", "pointsrcs","tod_shape","tod"]
def read(entry, fields=None, exclude=None, verbose=False):
	# Handle auto-stacking combo read transparently
	if isinstance(entry, list) or isinstance(entry, tuple):
		return read_combo(entry, fields=fields, exclude=exclude, verbose=verbose)
	# The normal case for a 1d to below
	if fields is None: fields = list(default_fields)
	if exclude is None: exclude = []
	for ex in exclude: fields.remove(ex)
	d = None
	for field in fields:
		t1 = time.time()
		if field is "tod" and d is not None:
			d2 = readers[field](entry, dets=d.dets)
		else:
			d2 = readers[field](entry)
		if d is None: d = d2
		else: d = dataset.merge([d,d2])
		t2 = time.time()
		if verbose: print "read  %-14s in %6.3f s" % (field, t2-t1)
	return d

def read_combo(entries, fields=None, exclude=None, verbose=False):
	# Read in each scan individually
	if fields is None: fields = list(default_fields)
	if exclude is None: exclude = []
	for ex in exclude: fields.remove(ex)
	# We need layout and boresight for combo reading
	if "layout" not in fields: fields = ["layout"] + fields
	if "boresight" not in fields: fields = ["boresight"] + fields
	ds = []
	for entry in entries:
		if verbose: print "reading %s" % entry.id
		ds.append(read(entry, fields=fields, verbose=verbose))
	if len(ds) == 1: return ds[0]
	# Offset samples to align them, and make detector ids unique
	det_offs = utils.cumsum([d.layout.ndet for d in ds])
	offs_real = measure_offsets([d.boresight[0] for d in ds])
	offs = np.round(offs_real).astype(int)
	assert np.all(np.abs(offs-offs_real) < 0.1), "Non-integer sample offset in read_combo"
	if verbose: print "offsets: " + ",".join([str(off) for off in offs])
	if verbose: print "shifting"
	for d, det_off, off in zip(ds, det_offs, offs):
		d.shift(det_off, off)
	# Find the common samples, as we must restrict to these before
	# we can take the union
	samples_list = np.array([d.samples for d in ds])
	samples = np.array([np.max(samples_list[:,0]),np.min(samples_list[:,1])])
	if verbose: print "restricting"
	for d in ds: d.restrict(samples=samples)
	# Ok, all datasets have the same sample range, and non-overlapping detectors.
	# Merge into a union dataset
	if verbose: print "union"
	dtot = dataset.detector_union(ds)
	# Detector layout cannot be automatically merged, so do it manually. We
	# assume that all have the same rectangular layout with the same number
	# of columns.
	row_offs = utils.cumsum([d.layout.nrow for d in ds])
	dtot.layout.rows = np.concatenate([d.layout.rows + off for d, off in zip(ds, row_offs)])
	dtot.layout.cols = np.concatenate([d.layout.cols for d in ds])
	dtot.layout.dark = np.concatenate([d.layout.dark for d in ds])
	dtot.layout.pcb  = np.concatenate([d.layout.pcb  for d in ds])
	dtot.layout.nrow = np.max(dtot.layout.rows)+1
	dtot.layout.ncol = np.max(dtot.layout.cols)+1
	dtot.layout.ndet = len(dtot.layout.rows)
	return dtot

def measure_offsets(times, nstep=10, dstep=1000, maxerr=0.1):
	"""Find the number of samples each timeseries in times[:,nsamp] is ahread
	of the first one."""
	times = np.array([t[0:dstep*nstep:dstep] for t in times])
	dt = np.median(times[:,1:]-times[:,:-1],1)/dstep
	# how many samples later each one starts compared to the first
	off = np.median(times-times[0],1)/dt
	return off

def require(data, fields):
	for field in fields:
		if field not in data:
			raise errors.DataMissing(field)

def calibrate_boresight(data):
	"""Calibrate the boresight by converting to radians and
	interpolating across missing samples linearly. Note that
	this won't give reasonable results for gaps of length
	similar to the scan period. Also adds a srate field containing
	the sampling rate."""
	require(data, ["boresight","flags"])
	# Convert angles to radians
	data.boresight[1:] = utils.unwind(data.boresight[1:] * np.pi/180)
	# Find unreliable regions
	bad = (data.flags!=0)*(data.flags!=0x10)
	#bad += srate_mask(data.boresight[0])
	# Interpolate through bad regions. For long regions, this won't
	# work, so these should be cut.
	#  1. Raise an exception
	#  2. Construct a cut on the fly
	#  3. Handle it in the autocuts.
	# The latter is cleaner in my opinion
	for b in data.boresight:
		gapfill.gapfill_linear(b, bad, inplace=True)
	srate = 1/utils.medmean(data.boresight[0,1:]-data.boresight[0,:-1])
	data += dataset.DataField("srate", srate)
	return data

def calibrate_hwp(data):
	"""Convert hwp from degrees to radians, and expand it to
	the full samples in data, since it might have dummy values if
	the hwp was not actually active for this tod."""
	require(data, ["hwp","hwp_id"])
	data.hwp = data.hwp * utils.degree
	if data.hwp_id == "none" and data.nsamp:
		del data.hwp
		hwp = np.zeros(data.nsamp)
		data += dataset.DataField("hwp", hwp, samples=[0,hwp.size], sample_index=0)
	return data

config.default("fft_factors", "2,3,5,7,11,13", "Crop TOD lengths to the largest number with only the given list of factors. If the list includes 1, no cropping will happen.")
def crop_fftlen(data, factors=None):
	"""Slightly crop samples in order to make ffts faster. This should
	be called at a point when the length won't be futher cropped by other
	effects."""
	if data.nsamp is None: raise errors.DataMissing("nsamp")
	factors = config.get("fft_factors", factors)
	if isinstance(factors, basestring): factors = [int(w) for w in factors.split(",")]
	ncrop = fft.fft_len(data.nsamp, factors=factors)
	data += dataset.DataField("fftlen", samples=[data.samples[0],data.samples[0]+ncrop])
	return data

def calibrate_point_offset(data):
	"""Convert pointing offsets from focalplane offsets to ra,dec offsets"""
	require(data, ["boresight", "point_offset"])
	data.point_offset[:] = offset_to_dazel(data.point_offset, np.mean(data.boresight[1:,::100],1))
	return data

def calibrate_polangle(data):
	"""Rotate polarization angles to match the Healpix convention"""
	require(data, ["polangle"])
	data.polangle += np.pi/2
	return data

config.default("pad_cuts", 0, "Number of samples by which to widen each cut range by")
def calibrate_cut(data, n=None):
	require(data, ["cut"])
	n = config.get("pad_cuts", n)
	data.cut = data.cut.widen(n)
	return data

def calibrate_beam(data):
	"""Make sure beam is equispaced. Convert radius to radians"""
	require(data, ["beam"])
	r, beam = data.beam
	assert r[0] == 0, "Beam must start from 0 radius"
	assert np.all(np.abs((r[1:]-r[:-1])/(r[1]-r[0])-1)<0.01), "Beam must be equispaced"
	data.beam = np.array([r*utils.degree, beam])
	return data

def calibrate_tod(data):
	"""Apply gain to tod and deconvolve instrument filters"""
	calibrate_tod_real(data)
	calibrate_tod_fourier(data)
	return data

config.default("gapfill", "copy", "TOD gapfill method. Can be 'copy' or 'linear'")
def calibrate_tod_real(data):
	"""Apply gain to tod, fill gaps and deslope"""
	require(data, ["tod","gain","cut"])
	#print data.tod.shape, data.samples
	#print data.dets[:4]
	#np.savetxt("test_enki1/tod_raw.txt", data.tod[0])
	data.tod = data.tod * data.gain[:,None]
	#np.savetxt("test_enki1/tod_gain.txt", data.tod[0])
	gapfiller = {"copy":gapfill.gapfill_copy, "linear":gapfill.gapfill_linear}[config.get("gapfill")]
	gapfiller(data.tod, data.cut, inplace=True)
	#np.savetxt("test_enki1/tod_gapfill.txt", data.tod[0])
	utils.deslope(data.tod, w=8, inplace=True)
	#np.savetxt("test_enki1/tod_deslope.txt", data.tod[0])
	return data

def calibrate_tod_fourier(data):
	"""Deconvolve instrument filters and time constants from TOD"""
	require(data, ["tod", "tau", "srate"])
	if data.tod.size == 0: return data
	ft     = fft.rfft(data.tod)
	freqs  = np.linspace(0, data.srate/2, ft.shape[-1])
	butter = filters.butterworth_filter(freqs)
	for di in range(len(ft)):
		ft[di] /= filters.tconst_filter(freqs, data.tau[di])*butter
	fft.irfft(ft, data.tod, normalize=True)
	#np.savetxt("test_enki1/tod_detau.txt", data.tod[0])
	del ft
	return data

# These just turn cuts on or off, without changing their other properties
config.default("cut_turnaround", False, "Whether to apply the turnaround cut.")
config.default("cut_ground",     False, "Whether to apply the turnaround cut.")
config.default("cut_sun",        False, "Whether to apply the sun distance cut.")
config.default("cut_moon",       False, "Whether to apply the moon distance cut.")
config.default("cut_pickup",     False, "Whether to apply the pickup cut.")
config.default("cut_stationary", True,  "Whether to apply the stationary ends cut")
config.default("cut_tod_ends",   True,  "Whether to apply the tod ends cut")
config.default("cut_mostly_cut", True,  "Whether to apply the mostly cut detector cut")
config.default("cut_point_srcs", False, "Whether to apply the point source cut")
# These cuts are always active, but can be effectively based on the parameter value
config.default("cut_max_frac",    0.50, "Cut whole tod if more than this fraction is autocut.")
config.default("cut_tod_mindur",  3.75, "Minimum duration of tod in minutes")
config.default("cut_tod_mindet",   100, "Minimum number of usable detectors in tod")
# These just modify the behavior of a cut. Most of these are in cuts.py
config.default("cut_sun_dist",    30.0, "Min distance to Sun in Sun cut.")
config.default("cut_moon_dist",   10.0, "Min distance to Moon in Moon cut.")
def autocut(d, turnaround=None, ground=None, sun=None, moon=None, max_frac=None, pickup=None):
	"""Apply automatic cuts to calibrated data."""
	ndet, nsamp = d.ndet, d.nsamp
	if not ndet or not nsamp: return d
	# Insert a cut into d if necessary
	if "cut" not in d:
		d += dataset.DataField("cut", rangelist.Multirange.empty(ndet,nsamp))
	# insert an autocut datafield, to keep track of how much data each
	# automatic cut cost us
	d += dataset.DataField("autocut", [])
	def addcut(label, dcut):
		n0, dn = d.cut.sum(), dcut.sum()
		d.cut = d.cut + dcut
		if isinstance(dcut, rangelist.Rangelist): dn *= ndet
		d.autocut.append([ label, dn, d.cut.sum() - n0 ]) # name, mycut, myeffect
	if config.get("cut_stationary") and "boresight" in d:
		addcut("stationary", cuts.stationary_cut(d.boresight[1]))
	if config.get("cut_tod_ends") and "srate" in d:
		addcut("tod_ends", cuts.tod_end_cut(nsamp, d.srate))
	if config.get("cut_turnaround", turnaround) and "boresight" in d:
		addcut("turnaround",cuts.turnaround_cut(d.boresight[0], d.boresight[1]))
	if config.get("cut_ground", ground) and "boresight" in d and "point_offset" in d:
		addcut("ground", cuts.ground_cut(d.boresight, d.point_offset))
	if config.get("cut_sun", sun) and "boresight" in d and "point_offset" in d and "site" in d:
		addcut("avoidance",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Sun", config.get("cut_sun_dist")*np.pi/180))
	if config.get("cut_moon", moon) and "boresight" in d and "point_offset" in d and "site" in d:
		addcut("moon",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Moon", config.get("cut_moon_dist")*np.pi/180))
	if config.get("cut_pickup", pickup) and "boresight" in d and "pickup_cut" in d:
		addcut("pickup",cuts.pickup_cut(d.boresight[1], d.dets, d.pickup_cut))
	if config.get("cut_mostly_cut"):
		addcut("mostly_cut", cuts.cut_mostly_cut_detectors(d.cut))
	if config.get("cut_point_srcs"):
		params = pointsrcs.src2param(d.pointsrcs)
		params[:,5:7] = 1
		params[:,7]   = 0
		c = cuts.point_source_cut(d, params)
		addcut("point_srcs", c)
	# What fraction is cut?
	cut_fraction = float(d.cut.sum())/d.cut.size
	# Get rid of completely cut detectors
	keep = np.where(d.cut.sum(flat=False) < nsamp)[0]
	d.restrict(d.dets[keep])

	def cut_all_if(label, condition):
		if condition: dcut = rangelist.Rangelist.ones(nsamp)
		else: dcut = rangelist.Rangelist.empty(nsamp)
		addcut(label, dcut)
	cut_all_if("max_frac",   config.get("cut_max_frac", max_frac) < cut_fraction)
	if "srate" in d:
		cut_all_if("tod_mindur", config.get("cut_tod_mindur") > nsamp/d.srate/60)
	cut_all_if("tod_mindet", config.get("cut_tod_mindet") > ndet)
	# Get rid of completely cut detectors again
	keep = np.where(d.cut.sum(flat=False) < nsamp)[0]
	d.restrict(d.dets[keep])

	return d

calibrators = {
	"boresight":    calibrate_boresight,
	"point_offset": calibrate_point_offset,
	"beam":         calibrate_beam,
	"polangle":     calibrate_polangle,
	"cut":          calibrate_cut,
	"autocut":      autocut,
	"fftlen":       crop_fftlen,
	"tod":          calibrate_tod,
	"tod_real":     calibrate_tod_real,
	"tod_fourier":  calibrate_tod_fourier,
	"hwp":          calibrate_hwp,
}

default_calib = ["boresight", "polangle", "hwp", "point_offset", "beam", "cut", "fftlen", "autocut", "tod"]
def calibrate(data, operations=None, exclude=None, strict=False, verbose=False):
	"""Calibrate the DataSet data by applying the given set of calibration
	operations to it in the given order. Data is modified inplace. If strict
	is True, then specifying a calibration operation that depends on a field
	that is not present in data raises a DataMissing exception. Otherwise,
	these are silently ignored. strict=False is useful for applying all applicable
	calibrations to a DataSet that only contains a subset of the data."""
	if operations is None: operations = list(default_calib)
	if exclude is None: exclude = []
	for ex in exclude: operations.remove(ex)
	for op in operations:
		t1 = time.time()
		status = 1
		try:
			calibrators[op](data)
		except errors.DataMissing as e:
			if strict: raise
			status = 0
		t2 = time.time()
		if verbose: print "calib %-14s in %6.3f s" % (op, t2-t1) + ("" if status else " [skipped]")
	return data

# Helper functions

# Rz(-az)Ry(pi/2-el)Rx(y)Ry(-x): What is the meaning of this operation?
#
# Ry(-x): Rotate detector pointing from (0,0,1) to about (x,0,1)
# Rx(y):  Rotate detector pointing from (x,0,1) to about (x,y,1)
# Ry(pi/2-el): Rotate from zenith to target elevation
# Rx(-az): Rotate to target azimuth
#
# That makes sense. x and y are angular coordinates in a focalplane
# centered coordinate system, with x,y being analogous to (dec,ra).
# So if hor = [az,el] and bore=[bore_az,bore_el], then
# [y,x] = coordinates.recenter(hor, np.concatenate([bore,bore*0]))
# Yes, this works - we understand focalplane coordinates.
#
# For beam simulations I really need detector-centered coordinates.
# Approximating these as xy - xy_det should be accurate to far
# higher accuracy than our beam is known.

def offset_to_dazel(offs, azel):
	"""Convert from focalplane offsets to offsets in horizontal coordinates.
	Corresponds to the rotation Rz(-az)Ry(pi/2-el)Rx(y)Ry(-x). Taken from
	libactpol. The previous version of this was equivalent in the flat sky
	limit, but for non-tiny angles deviated by several arcseconds.
	offs should be [dx,dy] according to the data file ordering, as
	returned by the files module."""
	az, el = azel
	x, y = np.asarray(offs).T
	# Formula below based on libactpol, which uses opposite
	# ordering of y and x, so swap
	y, x = x, y
	p = [ -np.sin(x), -np.cos(x)*np.sin(y), np.cos(x)*np.cos(y) ]
	p = [ np.sin(el)*p[0]+np.cos(el)*p[2], p[1], -np.cos(el)*p[0]+np.sin(el)*p[2] ]
	dEl = np.arcsin(p[2])-el
	dAz = -np.arctan2(p[1],p[0])
	return np.array([dAz,dEl]).T

def dazel_to_offset(dazel, azel):
	"""Inverse function of offset_to_dazel"""
	az, el = azel
	dazel = np.asarray(dazel).T
	p = np.array([np.cos(-dazel[0]),np.sin(-dazel[0]),np.sin(dazel[1]+el)])
	norm = ((1-p[2]**2)/(p[0]**2+p[1]**2))**0.5
	p[:2] *= norm
	p = [np.sin(el)*p[0]-np.cos(el)*p[2],p[1], np.sin(el)*p[2]+np.cos(el)*p[0]]
	x = np.arcsin(-p[0])
	y = np.arctan2(-p[1],p[2])
	y, x = x, y
	return np.array([x,y]).T
