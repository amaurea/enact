"""This module implements extra, dynamic data cuts for ACT data."""
import numpy as np, time
from enlib.resample import resample_bin
from enlib import rangelist, utils, config, coordinates, array_ops, pmat

config.default("cut_turnaround_step", 20, "Smoothing length for turnaround cut. Pointing will be downsampled by this number before acceleration is computed.")
config.default("cut_turnaround_lim",   5, "Acceleration threshold for turnaround cut in units of standard deviations of the acceleration.")
config.default("cut_turnaround_margin",1, "Margin for turnaround cut in units of the smoothing length. This will be added on each side of the acceleration-masked regions. In units of downsampled samples.")
def turnaround_cut(t, az, step=None, lim=None, margin=None):
	"""Cut samples where the telescope is accelerating."""
	step   = config.get("cut_turnaround_step",   step)
	lim    = config.get("cut_turnaround_lim",    lim)
	margin = config.get("cut_turnaround_margin", margin)
	t2, az2 = resample_bin(t[:2000], [1.0/step]), resample_bin(az, [1.0/step])
	dt = np.median(t2[1:]-t2[:-1])
	ddaz = (az2[2:]+az2[:-2]-2*az2[1:-1])/dt**2
	mask = np.abs(ddaz) > 2*np.std(ddaz)
	mask = np.abs(ddaz) > lim*np.std(ddaz[~mask])
	r = utils.mask2range(mask)
	r[:,0] = np.maximum(0,     r[:,0]-margin)
	r[:,1] = np.minimum(len(t),r[:,1]+margin)
	r *= step
	return rangelist.Rangelist(r, len(t))

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
		cuts.append(rangelist.Rangelist(mask_az&mask_el))
	return rangelist.Multirange(cuts)

def avoidance_cut(bore, det_offs, site, name_or_pos, margin):
	"""Cut samples that get too close to the specified object
	(e.g. "Sun" or "Moon") or celestial position ([ra,dec] in racians).
	Margin specifies how much to avoid the object by."""
	cmargin = np.cos(margin)
	mjd = utils.ctime2mjd(bore[0])
	obj_pos    = coordinates.interpol_pos("cel","hor",name_or_pos,mjd,site)
	obj_pos[0] = utils.rewind(obj_pos[0], bore[1])
	cosel      = np.cos(obj_pos[1])
	# Only cut if above horizon
	above_horizon = obj_pos[1]>0
	null_cut = rangelist.Multirange.empty(det_offs.shape[0], bore.shape[1])
	if np.all(~above_horizon): return null_cut
	# Find center of array, and radius
	arr_center = np.mean(det_offs,0)
	arr_rad    = np.max(np.sum((det_offs-arr_center)**2,0)**0.5)
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
		cuts.append(rangelist.Rangelist(det_mask_full))
	res = rangelist.Multirange(cuts)
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
	if np.all(~above_horizon): return rangelist.Multirange.empty(det_offs.shape[0], bore.shape[1])
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
		cuts.append(rangelist.Rangelist(bad))
	res = rangelist.Multirange(cuts)
	return res

def det2hex(dets, ncol=32):
	res = []
	for det in dets:
		col = det % ncol
		if   col < 24: hex = col/8
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
	res = rangelist.Multirange.empty(len(dets),len(az))
	for cdir,chex,az1,az2,strength in pcut:
		myrange  = rangelist.Rangelist((az>=az1)&(az<az2)&(dir==cdir))
		uncut    = rangelist.Rangelist.empty(len(az))
		mycut    = []
		for h in hex:
			if h == chex: mycut.append(myrange)
			else:         mycut.append(uncut)
		res = res + rangelist.Multirange(mycut)
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
		return rangelist.Rangelist.ones(len(az))
	return rangelist.Rangelist([[0,b1[0]],[b2[-1],len(az)]],len(az))

config.default("cut_tod_ends_nsec", 0.5, "Number of seconds to cut at each end of tod")
def tod_end_cut(nsamp, srate, cut_secs=None):
	"""Cut cut_secs seconds of data at each end of the tod"""
	ncut = int(config.get("cut_tod_ends_nsec",cut_secs)*srate)
	return rangelist.Rangelist([[0,ncut],[nsamp-ncut,nsamp]], nsamp)

max_frac   = config.default("cut_mostly_cut_frac",   0.20, "Cut detectors with a higher fraction of cut samples than this.")
max_nrange = config.default("cut_mostly_cut_nrange", 50, "Cut detectors with a larger number of cut ranges than this.")
def cut_mostly_cut_detectors(cuts, max_frac=None, max_nrange=None):
	"""Mark detectors with too many cuts or too large cuts as completely cut."""
	max_frac   = config.get("cut_mostly_cut_frac",   max_frac)
	max_nrange = config.get("cut_mostly_cut_nrange", max_nrange)
	cut_samps  = cuts.sum(flat=False)
	cut_nrange = np.array([len(c.ranges) for c in cuts.data])
	bad = (cut_samps > cuts.shape[-1]*max_frac) | (cut_nrange > max_nrange)
	ocuts = []
	for b in bad:
		if b: ocuts.append(rangelist.Rangelist.ones(cuts.shape[-1]))
		else: ocuts.append(rangelist.Rangelist.empty(cuts.shape[-1]))
	return rangelist.Multirange(ocuts)

config.default("cut_point_srcs_threshold", 20, "Signal threshold to use for point source cut. Areas where the source is straonger than this in uK will be cut.")
def point_source_cut(d, srcs, threshold=None):
	threshold = config.get("cut_point_srcs_threshold", threshold)
	# Sort-of-circular dependency here. I don't like
	# how actdata datasets are incompatible with scans.
	# Should I just replace scans with actdata objects?
	import actscan
	# Simulate sources
	tod  = np.zeros((d.ndet,d.nsamp), np.float32)
	srcs = srcs.astype(np.float64)
	scan = actscan.ACTScan(d.entry, d=d)
	psrc = pmat.PmatPtsrc2(scan, srcs)
	psrc.forward(tod, srcs)
	# Use them to define mask
	cuts = []
	for t in tod:
		cuts.append(rangelist.Rangelist(t > threshold))
	return rangelist.Multirange(cuts)

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
	return rangelist.Rangelist(bad)
