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
		dataset.DataField("cut", data, dets=dets, det_index=0, samples=samples, sample_index=1),
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

def read_layout(entry):
	data = try_read(files.read_layout, "layout", entry.layout)
	return dataset.DataSet([
		dataset.DataField("layout", data),
		dataset.DataField("entry", entry)])

def read_pointsrcs(entry):
	data = try_read(pointsrcs.read, "pointsrcs", entry.pointsrcs)
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
	if moby: dets, tod = try_read(files.read_tod_moby, "tod", entry.tod)
	else:    dets, tod = try_read(files.read_tod,      "tod", entry.tod)
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
		"tod": read_tod
	}

def read(entry, fields=["layout","beam","gain","polangle","tconst","cut","point_offsets","site","spikes","boresight","pointsrcs","tod_shape","tod"], verbose=False):
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

def crop_fftlen(data):
	"""Slightly crop samples in order to make ffts faster. This should
	be called at a point when the length won't be futher cropped by other
	effects."""
	if data.nsamp is None: raise errors.DataMissing("nsamp")
	ncrop = fft.fft_len(data.nsamp)
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

def calibrate_beam(data):
	"""Make sure beam is equispaced. Convert radius to radians"""
	require(data, ["beam"])
	r, beam = data.beam
	assert r[0] == 0, "Beam must start from 0 radius"
	assert np.all(np.abs((r[1:]-r[:-1])/(r[1]-r[0])-1)<0.01), "Beam must be equispaced"
	data.beam = np.array([r*utils.degree, beam])

def calibrate_tod(data):
	"""Apply gain to tod and deconvolve instrument filters"""
	calibrate_tod_real(data)
	calibrate_tod_fourier(data)
	return data

config.default("gapfill", "copy", "TOD gapfill method. Can be 'copy' or 'linear'")
def calibrate_tod_real(data):
	"""Apply gain to tod, fill gaps and deslope"""
	require(data, ["tod","gain","cut"])
	data.tod = data.tod * data.gain[:,None]
	gapfiller = {"copy":gapfill.gapfill_copy, "linear":gapfill.gapfill_linear}[config.get("gapfill")]
	gapfiller(data.tod, data.cut, inplace=True)
	utils.deslope(data.tod, w=8, inplace=True)
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
		d += dataset.DataField("cut", rangelist.empty([ndet,nsamp]))
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
	"autocut":      autocut,
	"fftlen":       crop_fftlen,
	"tod":          calibrate_tod,
	"tod_real":     calibrate_tod_real,
	"tod_fourier":  calibrate_tod_fourier,
}

def calibrate(data, operations=["boresight", "polangle", "point_offset", "beam", "fftlen", "autocut", "tod"], strict=False, verbose=False):
	"""Calibrate the DataSet data by applying the given set of calibration
	operations to it in the given order. Data is modified inplace. If strict
	is True, then specifying a calibration operation that depends on a field
	that is not present in data raises a DataMissing exception. Otherwise,
	these are silently ignored. strict=False is useful for applying all applicable
	calibrations to a DataSet that only contains a subset of the data."""
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
