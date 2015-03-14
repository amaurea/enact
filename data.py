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

# How to handle noise
#
# Having to regenerate noise files all the times is
# tedious, and does not extend well to signal extracted
# noise estimation since different input maps are needed for
# different tods. Generating a noise file takes about 10
# seconds for reading the tod and 10 seconds for the analysis,
# so precomputing does not save much time for the actual
# map-making.
#
# We will therefore do the following.
# 1. A configuration parameter specifies the noise model,
#    which can be "file", "jon", etc.
#    a) If file is specified, the value found in the filedb is used.
#    b) Otherwise, a noise model is generated in "calibrate"
# 2. If we want to do signal subtraction, that must be handled
#    elsewhere, where one knows more about the signal. In this
#    case the calling code may wish to indicate that no noise
#    estimation should be done, so as to avoid wasting time
#    computing an unnecessary model. We handle this using
#    the nonoise argument.
# A problem with this approach is that the noise model only can be
# estimated when tod samples are read. That means that the full
# TOD will need to be read in the ACTScan constructor, and then
# discarded only to be read again in get_samples() later. Not
# exactly optimal. On the other hand, the same thing effectively
# happens when running tod2nmat followed by tod2map.
#
# Perhaps a better approach is to simply run tod2nmat only on
# the set of files one actually cares about as a pre-run before
# tod2map. But then one needs a convenient way of specifying
# the location of these new files to tod2map. The current system
# requires one to edit a filedb.

import numpy as np
from enact import files, filters, cuts
from enlib import zgetdata, utils, gapfill, fft, errors, scan, nmat, resample, config
from bunch import Bunch # use a simple bunch for now

config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
#config.default("noise_model", "file", "Which noise model to use. Can be 'file' to read from the files indicated by the filedb, or the name of a specific noise mode such as 'jon'")
class ACTScan(scan.Scan):
	def __init__(self, entry, subdets=None, d=None):
		if d is None:
			d = read(entry, ["gain","polangle","tconst","cut","point_offsets","boresight","site","noise"], subdets=subdets)
			calibrate(d)
		ndet = d.polangle.size
		# Necessary components for Scan interface
		self.mjd0      = utils.ctime2mjd(d.boresight[0,0])
		self.boresight = np.ascontiguousarray(d.boresight.T.copy())
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
		self.sys = "hor"
		self.site = d.site
		self.noise = d.noise
		# Implementation details
		self.entry = entry
		self.subdets = np.arange(ndet)
		self.sampslices = []
	def get_samples(self):
		"""Return the actual detector samples. Slow! Data is read from disk and
		calibrated on the fly, so store the result if you need to reuse it."""
		d = read(self.entry, subdets=self.subdets)
		calibrate(d)
		tod = d.tod
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

def read(entry, fields=["gain","polangle","tconst","cut","point_offsets","tod","boresight","site","noise"], subdets=None):
	"""Given a filedb entry, reads all the data associated with the
	fields specified (default: ["gain","polangle","tconst","cut","point_offsets","tod","boresight","site"]).
	Only detectors for which all the information is present will be
	returned, and missing files will raise a DataMissing error."""
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
		if "point_offsets" in fields:
			reading = "point_offsets"
			dets.point_offset, res.point_offset  = files.read_point_template(entry.point_template)
			res.point_offset += files.read_point_offsets(entry.point_offsets)[entry.id]
		if "site" in fields:
			reading = "site"
			res.site = files.read_site(entry.site)
		if "noise" in fields:
			reading = "noise"
			res.noise  = nmat.read_nmat(entry.noise)
			dets.noise = res.noise.dets
	except IOError  as e: raise errors.DataMissing("%s [%s] [%s]" % (e.message, reading, entry.id))
	except KeyError as e: raise errors.DataMissing("Gain correction or pointing offset [%s]" % entry.id)
	# Restrict to common set of ids
	inds  = utils.dict_apply_listfun(dets, utils.common_inds)
	for key in dets:
		I = inds[key]
		if len(I) == 0: raise errors.DataMissing("All detectors rejected!")
		# Restrict to user-chosen subset. NOTE: This is based on indices
		# into the set that would normally be accepted, not raw detector
		# values!
		if subdets is not None: I = I[subdets]
		res[key]  = res[key][I]
		dets[key] = np.array(dets[key])[I]
	dets = dets.values()[0]
	# Then get the boresight and time-ordered data
	try:
		with zgetdata.dirfile(entry.tod) as dfile:
			if "boresight" in fields:
				res.boresight, res.flags = files.read_boresight(dfile)
			if "tod" in fields:
				dets, res.tod = files.read_tod(dfile, dets)
	except zgetdata.OpenError as e:
		raise errors.DataMissing(e.message + "[%s]" % entry.id)
	res.dets = dets
	# Fill in default sample offset if missing
	if "sample_offset" not in res:
		res.sample_offset = 0
		res.cutafter = min([res[a].shape[-1] for a in ["tod","boresight","flags"] if a in res])+res_sample_offset
	return res

def calibrate(data):
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
		bad = srate_mask(data.boresight[0]) + (data.flags!=0)*(data.flags!=0x10)
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

	# Apply gain, make sure cut regions are reasonably well-behaved,
	# and make it fourier-friendly by removing a slope.
	if "tod" in data:
		data.tod = data.tod * data.gain[:,None]
		gapfill.gapfill_copy(data.tod, data.cut, inplace=True)
		utils.deslope(data.tod, w=8, inplace=True)

		# Unapply instrument filters
		ft     = fft.rfft(data.tod)
		freqs  = np.linspace(0, data.srate/2, ft.shape[-1])
		butter = filters.butterworth_filter(freqs)
		for di in range(len(ft)):
			ft[di] /= filters.tconst_filter(freqs, data.tau[di])*butter
		fft.irfft(ft, data.tod, normalize=True)

	# Convert pointing offsets from focalplane offsets to ra,dec offsets
	if "point_offset" in data:
		data.point_offset = offset_to_dazel(data.point_offset, data.boresight[1:,0])

	#apply_extra_cuts(data)

	# We operate in-place, but return for good measure
	return data

def offset_to_dazel(offs, azel):
	az, el = azel
	dx, dy = np.asarray(offs).T
	dz = np.sqrt(1-dx**2-dy**2)
	y2 = dz*np.sin(el)+dy*np.cos(el)
	z2 = dz*np.cos(el)-dy*np.sin(el)
	dEl = np.arcsin(y2)-el
	dAz = np.arctan2(dx, z2)
	return np.array((dAz,dEl)).T

def dazel_to_offset(dazel, azel):
	az, el = azel
	da, de = np.asarray(dazel).T
	y2 = np.sin(el+de)
	dx = np.sin(da)*np.cos(el+de)
	z2 = dx/np.tan(da)
	dy = y2*np.cos(el)-z2*np.sin(el)
	return np.array((dx,dy)).T

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

config.default("cut_turnaround", False, "Whether to apply the internal turnaround cut")
config.default("cut_ground",     False, "Whether to apply the internal ground cut")
def apply_extra_cuts(data, do_turnaround=None, do_ground=None):
	c = data.cut
	if config.get("cut_turnaround", do_turnaround):
		c = c + cuts.turnaround_cut(data.boresight[0], data.boresight[1])
	if config.get("cut_ground", do_ground):
		c = c + cuts.ground_cut(data.boresight, data.point_offset)
	#c = c + cuts.test_cut(data.boresight)
	data.cut = c
