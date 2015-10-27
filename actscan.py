import numpy as np, time
from enact import nmat_measure, actdata
from enlib import utils, scan, nmat, resample, config

config.default("cut_noise_whiteness", False, "Whether to apply the noise_cut or not")
config.default("cut_spikes", True, "Whether to apply the spike cut or not")
config.default("downsample_method", "fft", "Method to use when downsampling the TOD")
config.default("noise_model", "file", "Which noise model to use. Can be 'file' or 'jon'")
config.default("tod_window", 0.0, "Number of samples to window the tod by on each end")
config.default("tod_skip_deconv", False, "Whether to skip the time constant and butterworth deconvolution in actscan")
class ACTScan(scan.Scan):
	def __init__(self, entry, subdets=None, d=None, verbose=False):
		self.fields = ["gain","polangle","tconst","cut","point_offsets","boresight","site","tod_shape","layout"]
		if config.get("noise_model") == "file":
			self.fields += ["noise"]
		else:
			if config.get("cut_noise_whiteness"):
				self.fields += ["noise_cut"]
			if config.get("cut_spikes"):
				self.fields += ["spikes"]
		if d is None:
			d = actdata.read(entry, self.fields, verbose=verbose)
			actdata.calibrate(d, verbose=verbose)
			d.restrict(dets=d.dets[subdets])
		ndet = d.ndet
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
		self.dgrid = (d.layout.nrow, d.layout.ncol)
		self.sys = "hor"
		self.site = d.site
		if "noise" in d:
			self.noise = d.noise
		else:
			spikes = d.spikes[:2].T if "spikes" in d else None
			self.noise = nmat_measure.NmatBuildDelayed(model = config.get("noise_model"), window=d.srate*config.get("tod_window"), spikes=None)
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
		self.sampslices = []
	def get_samples(self, verbose=False):
		"""Return the actual detector samples. Slow! Data is read from disk and
		calibrated on the fly, so store the result if you need to reuse it."""
		# Because we've read the tod_shape field earlier, we know that reading tod
		# won't cause any additional truncation of the samples or detectors.
		t1 = time.time()
		self.d += actdata.read_tod(self.entry, dets=self.d.dets)
		t2 = time.time()
		if verbose: print "read  %-14s in %6.3f s" % ("tod", t2-t1)
		if config.get("tod_skip_deconv"): ops = ["tod_real"]
		else: ops = ["tod"]
		actdata.calibrate(self.d, operations=ops, verbose=verbose)
		tod = self.d.tod
		# Remove tod from our local d, so we won't end up hauling it around forever
		del self.d.tod
		method = config.get("downsample_method")
		for s in self.sampslices:
			srange = slice(s.start, s.stop, np.sign(s.step) if s.step else None)
			tod = tod[:,srange]
			tod = resample.resample(tod, 1.0/np.abs(s.step or 1), method=method)
		tod = np.ascontiguousarray(tod)
		return tod
	def __repr__(self):
		return self.__class__.__name__ + "[ndet=%d,nsamp=%d,id=%s]" % (self.ndet,self.nsamp,self.entry.id)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.sampslices.append(sampslice)
		res.d.restrict(dets=res.d.dets[detslice])
		return res
