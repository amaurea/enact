from __future__ import division, print_function
import numpy as np, time
from enact import nmat_measure, actdata
from enlib import utils, scan, nmat, resample, config, errors, bench, sampcut

config.default("cut_noise_whiteness", False, "Whether to apply the noise_cut or not")
config.default("cut_spikes", True, "Whether to apply the spike cut or not")
config.default("tod_sys", "hor", "Coordinate system the TOD is in. 'hor': Ideal horizontal coordinates. 'tele': non-ideal telescope coordinates.")
config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
config.default("noise_model", "jon", "Which noise model to use. Can be 'file' or 'jon'")
config.default("tod_skip_deconv", False, "Whether to skip the time constant and butterworth deconvolution in actscan")
config.default("dummy_cut", 0.0, "Fraction of dummy cuts to inject *after* gapfilling and noise estimation")
config.default("dummy_cut_len", 1000, "Dummy cuts will be exponentially distributed up to this length")
class ACTScan(scan.Scan):
	def __init__(self, entry, subdets=None, d=None, verbose=False, dark=False):
		self.fields = ["gain","mce_filter","tags","polangle","tconst","hwp","cut","point_offsets","boresight","site","tod_shape","array_info","beam","pointsrcs", "buddies"]
		if dark: self.fields += ["dark"]
		if config.get("noise_model") == "file":
			self.fields += ["noise"]
		else:
			if config.get("cut_noise_whiteness"):
				self.fields += ["noise_cut"]
			if config.get("cut_spikes"):
				self.fields += ["spikes"]
		if d is None:
			d = actdata.read(entry, self.fields, verbose=verbose)
			d = actdata.calibrate(d, verbose=verbose)
			if subdets is not None:
				d.restrict(dets=d.dets[subdets])
		if d.ndet == 0 or d.nsamp == 0: raise errors.DataMissing("No data in scan")
		ndet = d.ndet
		# Necessary components for Scan interface
		self.mjd0      = utils.ctime2mjd(d.boresight[0,0])
		self.boresight = np.ascontiguousarray(d.boresight.T.copy()) # [nsamp,{t,az,el}]
		self.boresight[:,0] -= self.boresight[0,0]
		self.offsets   = np.zeros([ndet,self.boresight.shape[1]])
		self.offsets[:,1:] = d.point_offset
		self.cut       = d.cut.copy()
		self.cut_noiseest = d.cut_noiseest.copy()
		self.cut_basic = d.cut_basic
		self.comps     = np.zeros([ndet,4])
		self.beam      = d.beam
		self.pointsrcs = d.pointsrcs
		self.comps     = d.det_comps
		self.hwp = d.hwp
		self.hwp_phase = d.hwp_phase
		self.dets  = d.dets
		self.dgrid = (d.array_info.nrow, d.array_info.ncol)
		self.array_info = d.array_info
		self.sys = config.get("tod_sys", entry.tod_sys if "tod_sys" in entry else None)
		self.site = d.site
		self.speed = d.speed
		self.spikes = d.spikes[:2].T if "spikes" in d else None
		if "noise" in d:
			self.noise = d.noise
		else:
			self.noise = nmat_measure.NmatBuildDelayed(model = config.get("noise_model"), spikes=self.spikes,
					cut=self.cut_noiseest)
		if "dark_tod" in d:
			self.dark_tod = d.dark_tod
		if "dark_cut" in d:
			self.dark_cut = d.dark_cut
		if "buddy_comps" in d:
			# Expand buddy_offs to {dt,daz,ddec}
			self.buddy_comps = d.buddy_comps
			self.buddy_offs  = np.concatenate([d.buddy_offs[...,:1]*0,d.buddy_offs],-1)
		self.autocut = d.autocut if "autocut" in d else []
		# Implementation details. d is our DataSet, which we keep around in
		# because we need it to read tod consistently later. It will *not*
		# take part in any sample slicing operations, as that might make the
		# delayed tod read inconsistent with the rest. It could take part in
		# detector slicing as long as calibrate_tod operates on each detector
		# independently. This is true now, but would not be so if we did stuff
		# like common mode subtraction there. On the other hand, not doing this
		# would prevent slicing before reading from giving any speedup or memory
		# savings. I don't think allowing this should be a serious problem.
		self.d = d
		self.entry = entry
		def fmt_id(entry):
			if isinstance(entry, list): return "+".join([fmt_id(e) for e in entry])
			else:
				if entry.tag: return entry.id + ":" + entry.tag
				else: return entry.id
		self.id = fmt_id(entry)
		self.sampslices = []
		self.mapping = None

		# FIXME: debug test
		if config.get("dummy_cut") > 0:
			nmax  = int(config.get("dummy_cut_len"))
			# Power law between 1 and nmax, with slope -1.
			# C(w) = log(w)/log(nmax)
			# P(w) = w**-1/log(nmax)
			# w(C) = n**C
			# Mean: (nmax-1)/log(nmax)
			nmean = (nmax-1)/np.log(nmax)
			ncut = int(self.nsamp * config.get("dummy_cut") / nmean)
			cut_ranges = np.zeros([self.ndet, ncut, 2],int)
			w = (nmax**np.random.uniform(0, 1, size=[self.ndet, ncut])).astype(int)
			np.clip(w, 1, nmax)
			cut_ranges[:,:,0] = np.random.uniform(0, self.nsamp, size=[self.ndet, ncut]).astype(int)
			cut_ranges[:,:,0] = np.sort(cut_ranges[:,:,0],1)
			cut_ranges[:,:,1] = cut_ranges[:,:,0] + w
			np.clip(cut_ranges[:,:,1], 0, self.nsamp)
			cut_dummy = sampcut.from_list(cut_ranges, self.nsamp)
			print(np.mean(w), nmean, nmax, ncut)
			print("cut fraction before", float(self.cut.sum())/self.cut.size)
			self.cut *= cut_dummy
			print("cut fraction after", float(self.cut.sum())/self.cut.size)

	#def get_samples(self, verbose=False, debug_inject=None):
	def get_samples(self, verbose=False):
		"""Return the actual detector samples. Slow! Data is read from disk and
		calibrated on the fly, so store the result if you need to reuse it."""
		# Because we've read the tod_shape field earlier, we know that reading tod
		# won't cause any additional truncation of the samples or detectors.
		# tags is only needed here for read_combo support, but that is mostly broken
		# anyway.
		t1 = time.time()
		self.d += actdata.read(self.entry, fields=["tod", "tags"], dets=self.d.dets)
		#if debug_inject is not None: self.d.tod += debug_inject
		t2 = time.time()
		if verbose: print("read  %-14s in %6.3f s" % ("tod", t2-t1))
		if config.get("tod_skip_deconv"): ops = ["tod_real"]
		else: ops = ["tod"]
		actdata.calibrate(self.d, operations=ops, verbose=verbose)
		tod = self.d.tod
		# Remove tod from our local d, so we won't end up hauling it around forever
		del self.d.tod
		# HWP resample if needed
		if self.mapping is not None:
			tod = np.ascontiguousarray(utils.interpol(tod, self.mapping.oimap[None], order=1, mask_nan=False))
		method = config.get("downsample_method")
		for s in self.sampslices:
			srange = slice(s.start, s.stop, np.sign(s.step) if s.step else None)
			tod = tod[:,srange]
			# make sure we get exactly the same length the cuts will be expecting
			step= np.abs(s.step or 1)
			olen= (tod.shape[1]+step-1)//step
			tod = resample.resample(tod, float(olen)/tod.shape[1], method=method)
		tod = np.ascontiguousarray(tod)
		return tod
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d,id=%s]" % (self.ndet,self.nsamp,self.id)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.sampslices.append(sampslice)
		res.d.restrict(dets=res.d.dets[detslice])
		return res
	def __contains__(self, val):
		return val in self.__dict__
