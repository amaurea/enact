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
import numpy as np, enlib.slice
from enact import files, filters
from enlib import zgetdata, utils, gapfill, fft, errors, scan
from bunch import Bunch # use a simple bunch for now

class ACTScan(scan.Scan):
	def __init__(self, entry):
		d = read(entry, ["gain","polangle","tconst","cut","point_offsets","boresight","site"])
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
		self.comps[:,0] = 1
		self.comps[:,1] = np.cos(2*d.polangle)
		self.comps[:,2] = np.sin(2*d.polangle)
		self.comps[:,3] = 0
		self.sys = "hor"
		self.site = d.site
		# Implementation details
		self.entry = entry
		self.dets  = np.arange(ndet)
		self.sampslices = []
	def get_samples(self):
		"""Return the actual detector samples. Slow! Data is read from disk and
		calibrated on the fly, so store the result if you need to reuse it."""
		d = read(self.entry, subdets=self.dets)
		calibrate(d)
		tod = d.tod
		for s in self.sampslices:
			tod = enlib.slice.slice_downgrade(tod, s)
		return tod
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d,id=%s]" % (self.ndet,self.nsamp,self.entry.id)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.sampslices.append(sampslice)
		return res

def read(entry, fields=["gain","polangle","tconst","cut","point_offsets","tod","boresight","site"], subdets=None):
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
			dets.gain, res.gain = files.read_gain(entry.gain)
			res.gain *= files.read_gain_correction(entry.gain_correction)[entry.id]
		if "polangle" in fields:
			dets.polangle, res.polangle = files.read_polangle(entry.polangle)
		if "tconst" in fields:
			dets.tau,  res.tau = files.read_tconst(entry.tconst)
		if "cut" in fields:
			dets.cut, res.cut, res.sample_offset = files.read_cut(entry.cut)
		if "point_offsets" in fields:
			dets.point_offset, res.point_offset  = files.read_point_template(entry.point_template)
			res.point_offset += files.read_point_offsets(entry.point_offsets)[entry.id]
		if "site" in fields:
			res.site = files.read_site(entry.site)
	except IOError  as e: raise errors.DataMissing("%s [%s]" % (e.message, entry.id))
	except KeyError as e: raise errors.DataMissing("Gain correction or pointing offset [%s]" % entry.id)
	# Restrict to common set of ids
	inds  = utils.dict_apply_listfun(dets, utils.common_inds)
	for key in dets:
		I = inds[key]
		# Restrict to user-chosen subset. NOTE: This is based on indices
		# into the set that would normally be accepted, not raw detector
		# values!
		if subdets != None: I = I[subdets]
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
	return res

def calibrate(data):
	"""Prepares the data (in the format returned from data.read) for
	general consumption by applying calibration factors, deglitching,
	etc. Note: This function changes its argument."""
	# Apply the sample offset
	if "tod" in data:       data.tod = data.tod[:,data.sample_offset:]
	if "boresight" in data: data.boresight = data.boresight[:,data.sample_offset:]
	if "flags" in data:     data.flags = data.flags[data.sample_offset:]

	# Smooth over gaps in the encoder values and convert to radians
	if "boresight" in data:
		data.boresight[1:] = utils.unwind(data.boresight[1:] * np.pi/180)
		bad = srate_mask(data.boresight[0]) + (data.flags!=0)*(data.flags!=0x10)
		for b in data.boresight:
			gapfill.gapfill_linear(b, bad, inplace=True)

	# Apply gain, make sure cut regions are reasonably well-behaved,
	# and make it fourier-friendly by removing a slope.
	if "tod" in data:
		data.tod = data.tod * data.gain[:,None]
		gapfill.gapfill_copy(data.tod, data.cut, inplace=True)
		utils.deslope(data.tod, w=8, inplace=True)

		# Unapply instrument filters
		ft     = fft.rfft(data.tod)
		srate  = 1/utils.medmean(data.boresight[0,1:]-data.boresight[0,:-1])
		freqs  = np.linspace(0, srate/2, ft.shape[-1])
		butter = filters.butterworth_filter(freqs)
		for di in range(len(ft)):
			ft[di] /= filters.tconst_filter(freqs, data.tau[di])*butter
		fft.irfft(ft, data.tod)

	# Convert pointing offsets from focalplane offsets to ra,dec offsets
	if "point_offset" in data:
		data.point_offset = offset_to_radec(data.point_offset, data.boresight[1:,0])

	# We operate in-place, but return for good measure
	return data

def offset_to_radec(offs, azel):
	az, el = azel
	dx, dy = offs.T
	dz = np.sqrt(1-dx**2-dy**2)
	y2 = dz*np.sin(el)+dy*np.cos(el)
	z2 = dz*np.cos(el)-dy*np.sin(el)
	dEl = np.arcsin(y2)-el
	dAz = np.arctan2(dx, z2)
	return np.array((dAz,dEl)).T

def srate_mask(t, tolerance=0.5):
	"""Returns a boolean array indicating which samples
	of a supposedly constant step-size array do not follow
	the dominant step size. tolerance indicates the maximum
	fraction of the average step to allow."""
	n    = t.size
	dt   = utils.medmean(t[1:]-t[:-1])
	tmodel  = np.arange(t.size)*dt
	tmodel += utils.medmean(t-tmodel)
	return np.abs(t-tmodel)>dt*tolerance
