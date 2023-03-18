"""This module implements extra, dynamic data cuts for ACT data."""
from __future__ import division, print_function
import numpy as np, time
from scipy import ndimage, interpolate
from enlib.resample import resample_bin
from enlib import utils, config, coordinates, array_ops, pmat, sampcut

#config.default("cut_turnaround_lim",    7.00, "Acceleration threshold for turnaround cut in units of standard deviations of the acceleration.")
#config.default("cut_turnaround_step",   0.25, "Smoothing scale in seconds to use when computing acceleration")
#config.default("cut_turnaround_margin", 0.25, "Margin for turnaround cut in seconds.")
#def turnaround_cut(az, srate, lim=None, step=None, margin=None):
#	"""Cut samples where the telescope is accelerating."""
#	# First compute the smoothed azimuth acceleration. Smoothing is needed because
#	# the acceleration would be far too noisy otherwise
#	step   = config.get("cut_turnaround_step", step)
#	sampstep = int(np.round(step*srate))
#	samps  = np.arange(len(az))
#	knots  = samps[::sampstep][1:-1]
#	spline = interpolate.splrep(samps, az, t=knots)
#	ddaz   = interpolate.splev(samps, spline, 2)*srate**2
#	# Then find the typical acceleration noise, which we will use to
#	# define areas of significant acceleration
#	lim    = config.get("cut_turnaround_lim", lim)
#	addaz  = np.abs(ddaz)
#	sigma  = np.std(ddaz)
#	for i in range(3):
#		sigma = np.std(ddaz[addaz < sigma*4])
#	mask  = addaz > sigma*lim
#	# Build the cut, and grow it by the margin
#	cut   = sampcut.from_mask(mask)
#	margin= utils.nint(config.get("cut_turnaround_margin", margin)*srate/2)
#	cut   = cut.widen(margin)
#	return cut

# New, simpler turnaround cuts. Simply cuts a given number of degrees away from the
# extrema.
config.default("cut_turnaround_margin", 0.2, "Margin for turnaround cut in degrees.")
def turnaround_cut(az, margin=None):
	margin = config.get("cut_turnaround_margin", margin)*utils.degree
	# Use percentile just in case there's some outliers (for example a scan that's a bit
	# higher than the others.
	az1    = np.percentile(az,  0.1)
	az2    = np.percentile(az, 99.9)
	mask   = (az<az1)|(az>az2)
	cut    = sampcut.from_mask(mask)
	return cut

#	return res

config.default("cut_ground_az", "57:62,-62:-57,73:75", "Az ranges to consider for ground cut")
config.default("cut_ground_el", "0:38", "El ranges to consider for ground cut")
def ground_cut(bore, det_offs, az_ranges=None, el_ranges=None):
	az_ranges = np.array([[float(w) for w in tok.split(":")] for tok in config.get("cut_ground_az", az_ranges).split(",")])*np.pi/180
	el_ranges = np.array([[float(w) for w in tok.split(":")] for tok in config.get("cut_ground_el", el_ranges).split(",")])*np.pi/180
	n = bore.shape[1]
	cuts = []
	for di, doff in enumerate(det_offs):
		p = bore[1:]+doff[:,None]
		mask_az = np.full([n],False,dtype=bool)
		for ar in az_ranges:
			mask_az |= utils.between_angles(p[0], ar)
		mask_el = np.full([n],False,dtype=bool)
		for er in el_ranges:
			mask_el |= utils.between_angles(p[1], er)
		cuts.append(sampcut.from_mask(mask_az&mask_el))
	return sampcut.stack(cuts)

def avoidance_cut(bore, det_offs, site, name_or_pos, margin):
	"""Cut samples that get too close to the specified object
	(e.g. "Sun" or "Moon") or celestial position ([ra,dec] in radians).
	Margin specifies how much to avoid the object by."""
	cmargin = np.cos(margin)
	mjd = utils.ctime2mjd(bore[0])
	obj_pos    = coordinates.interpol_pos("cel","tele",name_or_pos,mjd,site)
	obj_pos[0] = utils.rewind(obj_pos[0], bore[1])
	cosel      = np.cos(obj_pos[1])
	# Only cut if above horizon
	above_horizon = obj_pos[1]>0
	null_cut = sampcut.empty(det_offs.shape[0], bore.shape[1])
	if np.all(~above_horizon): return null_cut
	# Find center of array, and radius
	arr_center = np.mean(det_offs,0)
	arr_rad    = np.max(np.sum((det_offs-arr_center)**2,1)**0.5)
	def calc_mask(det_pos, rad, mask=slice(None)):
		offs  = (det_pos-obj_pos[:,mask])
		offs[0] *= cosel[mask]
		dists2= np.sum(offs**2,0)
		return dists2 < rad**2
	# Find samples that could possibly be near object for any detector
	cand_mask  = calc_mask(arr_center[:,None] + bore[1:], margin+arr_rad)
	cand_mask &= above_horizon
	cand_inds  = np.where(cand_mask)[0]
	if len(cand_inds) == 0: return null_cut
	# Loop through all detectors and find out if each candidate actually intersects
	cuts = []
	for di, off in enumerate(det_offs):
		det_pos  = bore[1:,cand_inds]+off[:,None]
		det_mask = calc_mask(det_pos, margin, cand_inds)
		# Expand mask to full set
		det_mask_full = np.zeros(bore.shape[1], bool)
		det_mask_full[cand_inds] = det_mask
		# And use this to build actual cuts
		cuts.append(sampcut.from_mask(det_mask_full))
	res = sampcut.stack(cuts)
	return res

def avoidance_cut_old(bore, det_offs, site, name_or_pos, margin):
	"""Cut samples that get too close to the specified object
	(e.g. "Sun" or "Moon") or celestial position ([ra,dec] in racians).
	Margin specifies how much to avoid the object by."""
	cmargin = np.cos(margin)
	mjd = utils.ctime2mjd(bore[0])
	obj_pos  = coordinates.interpol_pos("cel","hor",name_or_pos,mjd,site)
	obj_rect = utils.ang2rect(obj_pos, zenith=False)
	# Only cut if above horizon
	above_horizon = obj_pos[1]>0
	if np.all(~above_horizon): return sampcut.empty(det_offs.shape[0], bore.shape[1])
	cuts = []
	for di, off in enumerate(det_offs):
		det_pos  = bore[1:]+off[:,None]
		#det_rect1 = utils.ang2rect(det_pos, zenith=False) # slow
		# Not sure defining an ang2rect in array_ops is worth it.. It's ugly,
		# (angle convention hard coded) and only gives factor 2 on laptop.
		det_rect = array_ops.ang2rect(np.ascontiguousarray(det_pos.T)).T
		cdist = np.sum(obj_rect*det_rect,0)
		# Cut samples above horizon that are too close
		bad  = (cdist > cmargin) & above_horizon
		cuts.append(sampcut.from_mask(bad))
	res = sampcut.stack(cuts)
	return res

def det2hex(dets, ncol=32):
	res = []
	for det in dets:
		col = det % ncol
		if   col < 24: hex = col//8
		elif col < 27: hex = 3
		elif col < 29: hex = 4
		else: hex = 5
		res.append(hex)
	return np.array(res)

def pickup_cut(az, dets, pcut):
	"""Cut samples as specified in the pcut struct, which works on hexed per
	tod per scanning direction."""
	hex = det2hex(dets)
	dir = np.concatenate([[0],az[1:]<az[:-1]])
	res = sampcut.empty(len(dets),len(az))
	for cdir,chex,az1,az2,strength in pcut:
		myrange  = sampcut.from_mask((az>=az1)&(az<az2)&(dir==cdir))
		uncut    = sampcut.empty(1, len(az))
		mycut    = []
		for h in hex:
			if h == chex: mycut.append(myrange)
			else:         mycut.append(uncut)
		res *= sampcut.stack(mycut)
	return res

config.default("cut_stationary_tol", 0.2, "Number of degrees the telescope must move before the scan is considered to have started. Also applies at the end of the tod.")
def stationary_cut(az, tol=None):
	"""Cut samples where the telescope isn't moving at the beginning
	and end of the tod."""
	tol = config.get("cut_stationary_tol", tol)*utils.degree
	b1 = np.where(np.abs(az-az[0])>tol)[0]
	b2 = np.where(np.abs(az-az[-1])>tol)[0]
	if len(b1) == 0 or len(b2) == 0:
		# Entire tod cut!
		return sampcut.full(1,len(az))
	else:
		return sampcut.from_list([[[0,b1[0]],[b2[-1],len(az)]]],len(az))

config.default("cut_tod_ends_nsec", 0.5, "Number of seconds to cut at each end of tod")
def tod_end_cut(nsamp, srate, cut_secs=None):
	"""Cut cut_secs seconds of data at each end of the tod"""
	ncut = int(config.get("cut_tod_ends_nsec",cut_secs)*srate)
	return sampcut.from_list([[[0,ncut],[nsamp-ncut,nsamp]]], nsamp)

max_frac   = config.default("cut_mostly_cut_frac",   0.20, "Cut detectors with a higher fraction of cut samples than this.")
max_nrange = config.default("cut_mostly_cut_nrange", 50, "Cut detectors with a larger number of cut ranges than this.")
def cut_mostly_cut_detectors(cuts, max_frac=None, max_nrange=None):
	"""Mark detectors with too many cuts or too large cuts as completely cut."""
	max_frac   = config.get("cut_mostly_cut_frac",   max_frac)
	max_nrange = config.get("cut_mostly_cut_nrange", max_nrange)
	cut_samps  = cuts.sum(axis=1)
	cut_nrange = cuts.nranges
	bad = (cut_samps > cuts.nsamp*max_frac)
	if max_nrange > 0:
		bad |= cut_nrange > max_nrange
	ocuts = []
	for b in bad:
		if b: ocuts.append(sampcut.full(1,cuts.nsamp))
		else: ocuts.append(sampcut.empty(1,cuts.nsamp))
	return sampcut.stack(ocuts)

def point_source_cut(d, srcs, thresholds=[]):
	if len(thresholds) == 0: return []
	# Sort-of-circular dependency here. I don't like
	# how actdata datasets are incompatible with scans.
	# Should I just replace scans with actdata objects?
	from . import actscan
	# Simulate sources
	tod  = np.zeros((d.ndet,d.nsamp), np.float32)
	srcs = srcs.astype(np.float64)
	scan = actscan.ACTScan(d.entry, d=d)
	psrc = pmat.PmatPtsrc(scan, srcs)
	psrc.forward(tod, srcs)
	# Use them to define mask
	cuts = [sampcut.from_mask(tod > tr) for tr in thresholds]
	return cuts

def tconst_cut(nsamp, taus, taumax):
	return sampcut.from_detmask(taus > taumax, nsamp)

def test_cut(bore, frac=0.3, dfrac=0.05):
	b  = bore[1:]
	db = np.median(np.abs(b[:,1:]-b[:,:-1]),1)
	si = np.argmax(db)
	r  = [np.min(b[si]),np.max(b[si])]
	w  = r[1]-r[0]
	c0 = r[0]+w*frac
	c  = [c0-w*dfrac,c0+w*dfrac]
	bad = (b[si]>=c[0])&(b[si]<c[1])
	if si == 1: bad[...] = False
	return sampcut.from_mask(bad)

def simple_glitch_cut(tod, bsize=20, scales=[4], tol=100, dblock=32):
	"""Look for dramatic glitches, processing each detector individually. Only meant for
	identifying nasty spikes that need to be gapfilled. scales controls the length
	scales considered. Something like [1,2,4,8,16] might be ideal for finding glitches
	with cosistent S/N on multiple length scales, but lower scales can take quite long.
	Let's try [4] for now. It takes about 2.5 seconds."""
	cuts = []
	ndet, nsamp = tod.shape
	for di1 in range(0, ndet, dblock):
		di2  = min(di1 + dblock, ndet)
		dtod = tod[di1:di2]
		bad  = np.zeros(dtod.shape, bool)
		for si, scale in enumerate(scales):
			wtod = utils.block_reduce(dtod, scale, np.mean)
			# Remove a linear trend every 100 samples. This should get rid of the atmosphere
			meds = utils.block_reduce(wtod, bsize, np.median)
			rtod = wtod - utils.block_expand(meds, bsize, wtod.shape[-1], "linear")
			# Compute the typical rms
			rms  = np.median(utils.block_reduce(rtod, bsize, np.std),-1)
			# Flag values that are much higher than this
			bad |= utils.block_expand(np.abs(rtod) > rms[:,None]*tol, scale, dtod.shape[-1], "nearest")
		cuts.append(sampcut.from_mask(bad))
	return sampcut.stack(cuts)
