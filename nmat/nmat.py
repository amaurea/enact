import numpy as np, enlib.nmat, enlib.fft
from nmat_core import nmat_core

class NmatDetvecs(enlib.nmat.NoiseMatrix):
	def __init__(self, params):
		self.bins = params.bins
		self.iNu  = params.iNu
		self.Q    = params.Q
		self.vbins= params.vbins
	def apply(self, tod):
		ft = enlib.fft.rfft(tod)
		fft_norm = tod.shape[1]
		nmat_core.nmat(ft.T, self.bins.T, self.iNu.T/fft_norm, self.Q.T/fft_norm**0.5, self.vbins.T)
		enlib.fft.irfft(ft, tod)
		return tod
	def white(self, tod):
		nmat_core.nwhite(tod.T, self.bins.T, self.iNu.T)
