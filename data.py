"""This module provides higher-level access to actpol data input, operating on
fildb entries instead of filenames, and reading all the information for you."""
import numpy as np
from enact import files, filters
from enlib import zgetdata, utils, gapfill, fft, errors
from bunch import Bunch # use a simple bunch for now

def read(entry, fields=["gain","polangle","tconst","cut","point_offsets","tod","boresight","site"]):
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
		res[key]  = res[key][inds[key]]
		dets[key] = np.array(dets[key])[inds[key]]
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
	data.tod = data.tod[:,data.sample_offset:]
	data.boresight = data.boresight[:,data.sample_offset:]
	data.flags = data.flags[data.sample_offset:]

	# Smooth over gaps in the encoder values and convert to radians
	data.boresight[1:] = utils.unwind(data.boresight[1:] * np.pi/180)
	bad = srate_mask(data.boresight[0]) + (data.flags!=0)*(data.flags!=0x10)
	for b in data.boresight:
		gapfill.gapfill_linear(b, bad, inplace=True)

	# Apply gain, make sure cut regions are reasonably well-behaved,
	# and make it fourier-friendly by removing a slope.
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
