"""This module implements extra, dynamic data cuts for ACT data."""
import numpy as np
from enlib.resample import resample_bin
from enlib import rangelist, utils, config

config.default("cut_turnaround_step", 20, "Smoothing length for turnaround cut")
config.default("cut_turnaround_lim",   3, "Acceleration threshold for turnaround cut")
config.default("cut_turnaround_margin",1, "Margin for turnaround cut in units of the smoothing length")
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

config.default("cut_ground_az", "58.75:60,-61:-59.75", "Az ranges to consider for ground cut")
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
