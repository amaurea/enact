"""This module provides definitions of the effective filters that
are applied to the TOD by the ACT hardware."""
import numpy as np

def tconst_filter(freq, f3db):
	"""Return the fourier space representation of the effect of
	detector time constants, for the given frequensies."""
	return 1/(2*np.pi*1j*freq*f3db+1)

def butterworth_filter(freq):
	"""Returns the fourier space representation of the
	Butterworth-like filter used in the ACT hardware
	at the frequenices specified."""
	f_raw = 1/(0.00000002*100*33)
	b = np.array([[-32092,15750],[-31238,14895]])*2.0**(-14)
	omega = 2*np.pi*freq/f_raw
	e1, e2 = np.exp(-1j*omega), np.exp(-2j*omega)
	tmp = (1+2*e1+e2)**2/(1+b[0,0]*e1+b[0,1]*e2)/(1+b[1,0]*e1+b[1,1]*e2)
	return tmp * (1+sum(b[0]))*(1+sum(b[1]))/16
