"""This module implements extra, dynamic data cuts for ACT data."""
import numpy as np, time
from enlib.resample import resample_bin
from enlib import rangelist, utils, config, coordinates, array_ops

config.default("cut_turnaround_step", 20, "Smoothing length for turnaround cut. Pointing will be downsampled by this number before acceleration is computed.")
config.default("cut_turnaround_lim",   3, "Acceleration threshold for turnaround cut in units of standard deviations of the acceleration.")
config.default("cut_turnaround_margin",1, "Margin for turnaround cut in units of the smoothing length. This will be added on each side of the acceleration-masked regions. In units of downsampled samples.")
def turnaround_cut(t, az, step=None, lim=None, margin=None):
	"""Cut samples where the telescope is accelerating."""
	step   = config.get("cut_turnaround_step",   step)
	lim    = config.get("cut_turnaround_lim",    lim)
	margin = config.get("cut_turnaround_margin", margin)
	t2, az2 = resample_bin(t[:2000], [1.0/step]), resample_bin(az, [1.0/step])
	dt = np.median(t2[1:]-t2[:-1])
	ddaz = (az2[2:]+az2[:-2]-2*az2[1:-1])/dt**2
	mask = np.abs(ddaz) > lim*np.std(ddaz)
	r = utils.mask2range(mask)
	r[:,0] -= margin
	r[:,1] += margin
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
