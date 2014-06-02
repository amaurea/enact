import numpy as np, enlib.nmat, enlib.fft, copy, enlib.slice
from nmat_core import nmat_core

class NmatDetvecs(enlib.nmat.NoiseMatrix):
	def __init__(self, params):
		self.bins = params.bins # [nbin,2]
		self.iNu  = params.iNu  # [nbin,ndet]
		self.Q    = params.Q    # [nvectot,ndet]
		self.vbins= params.vbins# [nbin]
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		fft_norm = tod.shape[1]
		nmat_core.nmat(ft.T, self.bins.T, self.iNu.T/fft_norm, self.Q.T/fft_norm**0.5, self.vbins.T)
		enlib.fft.irfft(ft, tod)
		return tod
	def white(self, tod):
		nmat_core.nwhite(tod.T, self.bins.T, self.iNu.T)
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		# Assume that our bins cover all frequencies (if they don't,
		# the result of applying the matrix won't have a consistent unit.
		nf1 = self.bins[-1,1]
		sampslice = enlib.slice.expand_slice(sampslice, nf1*2-1)
		nf2 = np.abs(sampslice.stop-sampslice.start)/2+1
		nf2_max = nf2/np.abs(sampslice.step)
		# We don't need to remove bins beyond nf2_max, since they will
		# be ignored by nmat_core anyway.
		res.bins = res.bins * nf2/nf1
		res.iNu = np.ascontiguousarray(res.iNu[:,detslice])
		res.Q   = np.ascontiguousarray(res.Q[:,detslice])
		return res
