"""This module provides higher-level access to actpol data input, operating on
fildb entries instead of filenames, and reading all the information for you.

These operations are all ACT-specific, but the results follow a more general interface
which could also be used for general simulations. What is actually needed for the
analysis is

 .boresight[nsamp,{t,az,el}]
 .point_offsets[ndet,{az,el}]
 .comps[ndet,:]
 .sys

for the left hand side, and additionally

 .tod[ndet,nsamp]

for the right hand side. Any function that provides this should be
processable by the higher-level functions. So instead of doing
 a = calibrate(read(entry))
 process(a)
you could do
 a = simulate(...)
 process(a)
"""

# Problem: How to read the actual TOD later than the rest of
# information, without compromising the rest of the information?
# For example, while ideally the TOD and boresight shoud have
# the sme length, they may be different. If the TOD is shorter,
# they must be made consistent somehow, and the easiest way it
# to truncate the boresight (and cuts). But that doesn't work
# if we don't know how long the tod is because we haven't read it
# yet!


import numpy as np
from bunch import Bunch
from enact import files, filters, cuts, nmat_measure
from enlib import zgetdata, utils, gapfill, fft, errors, scan, nmat, resample, config, pmat, rangelist

config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
config.default("gapfill", "copy", "TOD gapfill method. Can be 'copy' or 'linear'")
config.default("noise_model", "file", "Which noise model to use. Can be 'file' or 'jon'")
config.default("tod_window", 0.0, "Number of samples to window the tod by on each end")
class ACTScan(scan.Scan):
	def __init__(self, entry, subdets=None, d=None):
		self.fields = ["gain","polangle","tconst","cut","point_offsets","boresight","site"]
		if config.get("noise_model") == "file":
			self.fields += ["noise"]
		else:
			self.fields += ["spikes","noise_cut"]
		if d is None:
			d = read(entry, self.fields, subdets=subdets)
			calibrate(d)
			autocut(d)
		ndet = d.polangle.size
		# Necessary components for Scan interface
		self.mjd0      = utils.ctime2mjd(d.boresight[0,0])
		self.boresight = np.ascontiguousarray(d.boresight.T.copy()) # [nsamp,{t,az,el}]
		self.boresight[:,0] -= self.boresight[0,0]
		self.offsets   = np.zeros([ndet,self.boresight.shape[1]])
		self.offsets[:,1:] = d.point_offset
		self.cut       = d.cut.copy()
		self.comps     = np.zeros([ndet,4])
		# negative U component because this is the top row of a positive
		# rotation matrix [[c,-s],[s,c]].
		self.comps[:,0] = 1
		self.comps[:,1] = np.cos(+2*d.polangle)
		self.comps[:,2] = np.sin(-2*d.polangle)
		self.comps[:,3] = 0
		self.dets  = d.dets
		self.dgrid = (d.nrow, d.ncol)
		self.sys = "hor"
		self.site = d.site
		try:
			self.noise = d.noise
		except AttributeError:
			self.noise = nmat_measure.NmatBuildDelayed(model = config.get("noise_model"), window=d.srate*config.get("tod_window"), spikes=d.spikes[:2].T)
		try: self.autocut = d.autocut
		except KeyError: self.autocut = []
		# Implementation details
		self.entry = entry
		self.subdets = np.arange(ndet)
		self.sampslices = []
	def get_samples(self):
		"""Return the actual detector samples. Slow! Data is read from disk and
		calibrated on the fly, so store the result if you need to reuse it."""
		# It's important that this is done consistently with the initialization.
		# The current method does not really ensure that, since
		#  1. in theory the avilable dets might be different when tod is added to the fields
		#  2. When being passed a d we can't ensure that it's been cut consistently
		#  3. The result of autocut may depend on self.subdets
		# Also, autocuts should ideally happen *before* gapfilling, desloping, etc.
		d = read(self.entry, self.fields + ["tod"], subdets=self.subdets)
		calibrate(d)
		autocut(d)
		tod = d.tod
		del d.tod
		method = config.get("downsample_method")
		for s in self.sampslices:
			tod = resample.resample(tod, 1.0/np.abs(s.step or 1), method=method)
			s = slice(s.start, s.stop, np.sign(s.step) if s.step else None)
			tod = tod[:,s]
		res = np.ascontiguousarray(tod)
		return res
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d,id=%s]" % (self.ndet,self.nsamp,self.entry.id)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.sampslices.append(sampslice)
		res.subdets = res.subdets[detslice]
		return res

# I want to be able to do this:
#  1. d = read(entry, fields=[blah blah, but not tod])
#  2. d = read(entry, fields=["tod"], d=d) # adds tod to d
# or
#  1. d = read(entry, fields=[etc])
#  2. d = calibrate(d)
#  3. d = autocut(d)
#  4. d = hum, get consistently calibrared and cut tod also
# Might have to settle for a less general one, which singles out
# tod:
#  1. d = read(entry, fields=[...])
#  2. d = calibrate(d)
#  3. d = autocut(d)
#  4. tod = read_tod(d)
#  5. tod = calibrate_tod(tod, d)
# This would work, but is ugly in that it would yield different
# results from read, calibrate and autocut with the tod read from
# the beginning, due to the order in which calibration and autocuts happen.
# Basically, there are three steps:
#  1. Most of the calibration (affects both tod and other things)
#  2. Autocuts, which depend on calibration
#  3. Calibration that depends on cuts (this one only affects the tod)


def read(entry, fields=["gain","polangle","tconst","cut","point_offsets","tod","boresight","site","noise"], subdets=None, absdets=None, moby=False, _return_nfull=False):
	"""Given a filedb entry, reads all the data associated with the
	fields specified (default: ["gain","polangle","tconst","cut","point_offsets","tod","boresight","site"]).
	Only detectors for which all the information is present will be
	returned, and missing files will raise a DataMissing error."""
	if isinstance(entry, list):
		return read_combo(entry, fields=fields, subdets=subdets, absdets=absdets, moby=moby)
	keymap = {"gain": ["gain","gain_correction"], "point_offsets": ["point_template","point_offsets"], "boresight": ["tod"] }
	for key in fields:
		if key in keymap:
			for subkey in keymap[key]:
				if entry[subkey] is None:
					raise errors.DataMissing("Missing %s (needed for %s) in entry for %s" % (subkey,key,entry.id))
		else:
			if entry[key] is None:
				raise errors.DataMissing("Missing %s in entry for %s" % (key,entry.id))
	res, dets = Bunch(entry=entry), Bunch()
	try:
		# Perform all the scary read operations
		if "gain" in fields:
			reading = "gain"
			dets.gain, res.gain = files.read_gain(entry.gain)
			mask = np.isfinite(res.gain)*(res.gain != 0)
			dets.gain, res.gain = dets.gain[mask], res.gain[mask]
			reading = "gain correction"
			res.gain *= files.read_gain_correction(entry.gain_correction)[entry.id]
		if "polangle" in fields:
			reading = "plangle"
			dets.polangle, res.polangle = files.read_polangle(entry.polangle)
		if "tconst" in fields:
			reading = "tconst"
			dets.tau,  res.tau = files.read_tconst(entry.tconst)
		if "cut" in fields:
			reading = "cut"
			dets.cut, res.cut, res.sample_offset = get_cuts([entry.cut])
			res.cutafter = res.sample_offset + res.cut[0].n if len(dets.cut) > 0 else 0
			if "pickup_cut" in entry:
				try: res.pickup_cut = files.read_pickup_cut(entry.pickup_cut)[entry.id]
				except KeyError: pass
		if "point_offsets" in fields:
			reading = "point_offsets"
			dets.point_offset, res.point_offset  = files.read_point_template(entry.point_template)
			reading = "point correction"
			res.point_offset += files.read_point_offsets(entry.point_offsets)[entry.id]
		if "site" in fields:
			reading = "site"
			res.site = files.read_site(entry.site)
		if "noise" in fields:
			reading = "noise"
			res.noise = nmat.read_nmat(entry.noise)
			dets.noise = res.noise.dets
		if "noise_cut" in fields:
			reading = "noise_cut"
			dets.noise_cut = files.read_noise_cut(entry.noise_cut)[entry.id]
			res.noise_cut = np.arange(len(dets.noise_cut)) # dummy
		if "spikes" in fields:
			reading = "spikes"
			res.spikes = files.read_spikes(entry.spikes)
	except (IOError, KeyError) as e: raise errors.DataMissing("%s [%s] [%s]" % (e.message, reading, entry.id))
	# Restrict to common set of ids. If absdets is specified, then
	# restrict to detectors mentioned there.
	if len(dets) > 0:
		res, dets, nfull = restrict_to_common_dets(res, dets, subdets=subdets, absdets=absdets)
	# Then get the boresight and time-ordered data
	try:
		if moby:
			if "boresight" in fields:
				res.boresight, res.flags = files.read_boresight_moby(entry.tod)
			if "tod" in fields:
				dets, res.tod = files.read_tod_moby(entry.tod, dets)
		else:
			with zgetdata.dirfile(entry.tod) as dfile:
				if "boresight" in fields:
					res.boresight, res.flags = files.read_boresight(dfile)
				if "tod" in fields:
					dets, res.tod = files.read_tod(dfile, dets)
	except (zgetdata.OpenError,IOError) as e:
		raise errors.DataMissing("Error opening dirfile: " + e.message + "[%s]" % entry.id)
	res.dets = dets
	# Fill in default sample offset if missing
	if "sample_offset" not in res:
		res.sample_offset = 0
		res.cutafter = min([res[a].shape[-1] for a in ["tod","boresight","flags"] if a in res])+res.sample_offset
	res.nrow, res.ncol = 33, 32
	if _return_nfull:
		return res, nfull
	else:
		return res

def restrict_to_common_dets(res, dets, subdets=None, absdets=None):
	"""Given res:{key->arr[ndet]} and dets:{key->dets[ndet]}, 
	extract slices of each arr corresponding to the interesction
	of all the detector sets. Returns the modified res and the
	resulting detector ids, as well as the number of detectors
	prior to applying the optional subdet slicing."""
	if absdets is not None:
		dets.absdets = absdets
		res.absdets = np.arange(len(absdets))
	inds = utils.dict_apply_listfun(dets, utils.common_inds)
	nfull = None
	for key in dets:
		I = inds[key]
		nfull = len(I)
		if len(I) == 0: raise errors.DataMissing("All detectors rejected!")
		# Restrict to user-chosen subset. NOTE: This is based on indices
		# into the set that would normally be accepted, not raw detector
		# values!
		if subdets is not None: I = I[subdets[subdets<len(I)]]
		res[key]  = res[key][I]
		dets[key] = np.array(dets[key])[I]
	dets = dets.values()[0]
	return res, dets, nfull

def read_combo(entries, fields=["gain","polangle","tconst","cut","point_offsets","tod","boresight","site"], subdets=None, absdets=None, moby=False, start_tol=100, align_tol=utils.arcsec):
	if "noise" in fields: raise ValueError("read_combo doesn'ts upport reading noise from file")
	#if absdets is not None: raise NotImplementedError, "read_combo does not support absdets"
	#if subdets is not None: raise NotImplementedError, "read_combo does not support subdets"
	ds = []
	exceptions = []
	j = 0
	for entry in entries:
		try:
			d, nfull = read(entry, fields, subdets=subdets, absdets=absdets, moby=moby, _return_nfull=True)
			ds.append(d)
			if subdets is not None:
				subdets = np.array(subdets) - nfull
				subdets = np.array(subdets[subdets>=0])
			if absdets is not None:
				absdets = np.array(absdets) - j
			j += d.nrow*d.ncol
		except errors.DataMissing as e:
			exceptions.append(e)
	if len(ds) == 1: return ds[0]
	if len(ds) == 0: raise errors.DataMissing("; ".join([e.message for e in exceptions]))

	# Find the offset between each timeseries in unit of samples
	nstep, dstep = 10, 1000
	ts = np.array([d.boresight[0,0:dstep*nstep:dstep] for d in ds])
	dt = np.median(ts[:,1:]-ts[:,:-1],1)/dstep
	off = np.median(ts-ts[0],1)/dt # how many samples later each one starts compared to d[0]
	offi = np.round(off).astype(int)
	assert np.all(np.abs(off-offi) < 0.1), "Non-integer sample offset between entries in read_combo"
	# Sort them such that d[0] is the one that starts first.
	sorti = np.argsort(offi)
	ds   = [ds[i] for i in sorti]
	offi = offi[sorti]-np.min(offi)
	assert np.max(offi) < start_tol, "Starting times exceed tolerance (%f > %f)" % (np.max(offi), start_tol)
	# Check that the same offset applied to the other entries of the
	# boresight match up. offi is how much d[0] must be offset by to match the others.
	diffs = np.array([d.boresight[:,0:nstep*dstep:dstep]-ds[0].boresight[:,i:i+nstep*dstep:dstep] for d, i in zip(ds, offi)])
	assert np.std(diffs[1:]) < align_tol, "Failed to align boresights. Misalignment: %s" % (str(np.std(diffs[1:],1))/utils.arcsec)

	# Great! It's possible to make everything align. Figure out how long
	# the final tod will be, after padding
	nsamps = np.array([d.cutafter for d in ds])
	nsamp  = np.max(nsamps+offi)
	npad   = nsamp-nsamps

	# Ok, start building the output. First the basics
	res  = Bunch(entry=ds[0].entry, ncol=ds[0].ncol, nrow=sum([d.nrow for d in ds]))
	res.dets, j = [], 0
	for d in ds:
		res.dets.append(d.dets + j)
		j += d.nrow*d.ncol
	res.dets = np.concatenate(res.dets)
	det_offs = utils.cumsum([len(d.dets) for d in ds], endpoint=True)

	# Merge sample offsets. We will use the lest restrictive offset relative
	# tot he new start.
	sample_offsets  = np.array([d.sample_offset for d in ds])
	res.sample_offset = np.min(sample_offsets + offi)
	res.cutafter = nsamp

	# Cuts must be offset and padded. offi indicates how far each
	# d is from the start of the output d. Because we are changing
	# sample_offsets, we must renumber the ranges and add cuts
	# at the beginning
	if "cut" in fields:
		rs = []
		for i, d in enumerate(ds):
			multi = d.cut
			for rlist in multi.data:
				rnew = rlist.copy()
				myoff = offi[i] + sample_offsets[i] - res.sample_offset
				rnew.ranges += myoff
				rnew.n = nsamp - res.sample_offset
				rnew = rnew + [[0,myoff], [myoff+rlist.n,rnew.n]]
				rs.append(rnew)
		res.cut = rangelist.Multirange(rs)

	if "boresight" in fields:
		borelen  = np.max(np.array([d.boresight.shape[1] for d in ds])+offi)
		res.boresight = np.zeros([3,borelen], dtype=ds[0].boresight.dtype)
		flaglen  = np.max(np.array([d.flags.shape[0] for d in ds])+offi)
		res.flags= np.zeros([flaglen],dtype=ds[0].flags.dtype)
		for i, d in enumerate(ds):
			res.boresight[:,offi[i]:offi[i]+d.boresight.shape[1]] = d.boresight
			res.flags[offi[i]:offi[i]+d.flags.shape[0]] = d.flags

	if "tod" in fields:
		todlen  = np.max(np.array([d.tod.shape[1] for d in ds])+offi)
		res.tod = np.zeros([len(res.dets),todlen],ds[0].tod.dtype)
		for i, d in enumerate(ds):
			res.tod[det_offs[i]:det_offs[i+1],offi[i]:offi[i]+d.tod.shape[1]] = d.tod

	if "gain" in fields: res.gain = np.concatenate([d.gain for d in ds])
	if "polangle" in fields: res.polangle = np.concatenate([d.polangle for d in ds])
	if "tconst" in fields: res.tau = np.concatenate([d.tau for d in ds])
	if "point_offsets" in fields: res.point_offset = np.concatenate([d.point_offset for d in ds], axis=0)
	if "site" in fields: res.site = ds[0].site
	if "spikes" in fields: res.spikes = ds[0].spikes # not accurate

	return res

def calibrate(data, nofft=False):
	"""Prepares the data (in the format returned from data.read) for
	general consumption by applying calibration factors, deglitching,
	etc. Note: This function changes its argument."""
	# Apply the sample offset
	if "tod" in data:
		data.tod = data.tod[:,data.sample_offset:data.cutafter]
		nsamp = data.tod.shape[1]
	if "boresight" in data:
		data.boresight = data.boresight[:,data.sample_offset:data.cutafter]
		nsamp = data.boresight.shape[1]
	if "flags" in data:
		data.flags = data.flags[data.sample_offset:data.cutafter]
		nsamp = data.flags.shape[0]

	# Smooth over gaps in the encoder values and convert to radians
	if "boresight" in data:
		data.boresight[1:] = utils.unwind(data.boresight[1:] * np.pi/180)
		bad = (data.flags!=0)*(data.flags!=0x10)
		#bad += srate_mask(data.boresight[0])
		if np.sum(bad) > 0.1*len(bad):
			raise errors.DataMissing("Too many pointings flagged bad")
		for b in data.boresight:
			gapfill.gapfill_linear(b, bad, inplace=True)
		data.srate = 1/utils.medmean(data.boresight[0,1:]-data.boresight[0,:-1])

	# Truncate to a fourier-friendly length. This may cost up to about 1% of
	# the data. This is most naturally done here because
	#  1. We need to fft when calibrating the tod
	#  2. The desloping needs to be done on the final array.
	# A disadvantage is that data.cut will have a different
	# length if is the only thing read, compared to if it is
	# read together with tod, boresight or flags. This is because
	# cuts does not know the length of the data. We could mitigate
	# this by truncating at the end rather than beginning. But the
	# beginning has more systematic errors, so it is the best one
	# to remove.
	# Cutting the start was too error prone when the different arrays
	# can have different length. So went for cutting the end after all.
	nsamp = fft.fft_len(nsamp)
	if "tod"       in data: data.tod       = data.tod[:,:nsamp]
	if "boresight" in data: data.boresight = data.boresight[:,:nsamp]
	if "flags"     in data: data.flags     = data.flags[:nsamp]
	if "cut"       in data: data.cut       = data.cut[:,:nsamp]

	# Ideally autocuts would happen *here*

	# Apply gain, make sure cut regions are reasonably well-behaved,
	# and make it fourier-friendly by removing a slope.
	if "tod" in data:
		data.tod = data.tod * data.gain[:,None]
		gapfiller = {"copy":gapfill.gapfill_copy, "linear":gapfill.gapfill_linear}[config.get("gapfill")]
		gapfiller(data.tod, data.cut, inplace=True)
		utils.deslope(data.tod, w=8, inplace=True)
		#nmat.apply_window(data.tod, data.srate*config.get("tod_window"))

		# Unapply instrument filters
		if not nofft:
			ft     = fft.rfft(data.tod)
			freqs  = np.linspace(0, data.srate/2, ft.shape[-1])
			butter = filters.butterworth_filter(freqs)
			for di in range(len(ft)):
				ft[di] /= filters.tconst_filter(freqs, data.tau[di])*butter
			fft.irfft(ft, data.tod, normalize=True)
			del ft

	# Convert pointing offsets from focalplane offsets to ra,dec offsets
	if "point_offset" in data:
		data.point_offset = offset_to_dazel(data.point_offset, np.mean(data.boresight[1:,::100],1))

	# Match the Healpix polarization convention
	if "polangle" in data:
		data.polangle += np.pi/2

	if "pickup_cut" in data:
		data.pickup_cut = [[dir,hex,az1*np.pi/180,az2*np.pi/180,strength] for dir,hex,az1,az2,strength in data.pickup_cut]

	# We operate in-place, but return for good measure
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
	ndet, nsamp = d.cut.shape
	d.autocut = [] # list, because order matters
	def addcut(label, dcut):
		n0, dn = d.cut.sum(), dcut.sum()
		d.cut = d.cut + dcut
		if isinstance(dcut, rangelist.Rangelist): dn *= ndet
		d.autocut.append([ label, dn, d.cut.sum() - n0 ]) # name, mycut, myeffect
	if config.get("cut_stationary"):
		addcut("stationary", cuts.stationary_cut(d.boresight[1]))
	if config.get("cut_tod_ends"):
		addcut("tod_ends", cuts.tod_end_cut(nsamp, d.srate))
	if config.get("cut_turnaround", turnaround):
		addcut("turnaround",cuts.turnaround_cut(d.boresight[0], d.boresight[1]))
	if config.get("cut_ground", ground):
		addcut("ground", cuts.ground_cut(d.boresight, d.point_offset))
	if config.get("cut_sun", sun):
		addcut("avoidance",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Sun", config.get("cut_sun_dist")*np.pi/180))
	if config.get("cut_moon", moon):
		addcut("moon",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Moon", config.get("cut_moon_dist")*np.pi/180))
	if config.get("cut_pickup", pickup) and "pickup_cut" in d:
		addcut("pickup",cuts.pickup_cut(d.boresight[1], d.dets, d.pickup_cut))
	if config.get("cut_mostly_cut"):
		addcut("mostly_cut", cuts.cut_mostly_cut_detectors(d.cut))
	# What fraction is cut?
	cut_fraction = float(d.cut.sum())/d.cut.size
	# Get rid of completely cut detectors
	print "A"
	printshapes(d)
	keep = np.where(d.cut.sum(flat=False) < nsamp)[0]
	d = apply_det_slice(d, keep)
	print "B"
	printshapes(d)

	def cut_all_if(label, condition):
		if condition: dcut = rangelist.Rangelist.ones(nsamp)
		else: dcut = rangelist.Rangelist.empty(nsamp)
		addcut(label, dcut)
	cut_all_if("max_frac",   config.get("cut_max_frac", max_frac) < cut_fraction)
	cut_all_if("tod_mindur", config.get("cut_tod_mindur") > nsamp/d.srate/60)
	cut_all_if("tod_mindet", config.get("cut_tod_mindet") > ndet)
	keep = np.where(d.cut.sum(flat=False) < nsamp)[0]
	d = apply_det_slice(d, keep)
	print "C"
	printshapes(d)

	return d

def printshapes(d):
	for key in d:
		try:
			print "%s: %s" % (key, str(d[key].shape))
		except: pass

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

def srate_mask(t, tolerance=400*10.0):
	"""Returns a boolean array indicating which samples
	of a supposedly constant step-size array do not follow
	the dominant step size. tolerance indicates the maximum
	fraction of the average step to allow."""
	w    = 100
	t0   = t[0]
	dt   = (t[-1]-t[0])/(len(t)-1)
	# Build constant srate model and flag too large deviations
	tmodel  = np.arange(t.size)*dt+t0
	return np.abs(t-tmodel)>dt*tolerance

def get_cuts(fnames):
	for fname in fnames:
		try:
			return files.read_cut(fname)
		except IOError:
			continue
	raise IOError(str(fnames))

def apply_det_slice(d, sel):
	for key in ["tod", "tau", "point_offset", "cut", "polangle", "gain", "noise", "dets", "noise_cut"]:
		if hasattr(d, key): setattr(d, key, getattr(d, key)[sel])
	return d

def group_ids(ids, tol=10):
	# Match based on the first number in group
	times  = np.array([int(id[:id.index(".")]) for id in ids])
	inds   = np.argsort(times)
	stimes= times[inds]
	# Find equal ranges
	diffs = stimes[1:]-stimes[:-1]
	edges = np.concatenate([[0],np.where(diffs > tol)[0]+1,[len(stimes)]])
	groups= np.array([edges[:-1],edges[1:]]).T
	res = [[ids[inds[i]] for i in range(*group)] for group in groups]
	return res
