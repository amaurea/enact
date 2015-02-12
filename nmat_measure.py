import numpy as np, scipy as sp, enlib.bins
from enlib import nmat

# This is an implementation of the standard ACT noise model,
# which decomposes the noise into a detector-uncorrelated
# component (Nd) and a detector-correlated component
# (Nv = U'SU, with U being orthonormal). Nd and S are both
# constant within frequency bins, but are allowed to vary
# between them.
#
# The set of correlated noise modes (U) are determined
# by extracting the dominant eigenvectors of the measured
# noise covariance for each bin, where the preliminary
# U-modes from before that mode have already been projected
# out. This ensures that the final set U is orthogonal.
#
# The full set of parameters we need to determine, then,
# are Nd[nbin,ndet], U[nmode,ndet], S[nbin,nmode],
# and bins[nbin,2].
#
# Noise will be measured in units of uK sqrt(sample)
# (or the square of that for power, which is what
# we measure here), in time domain. We want compatible
# units in frequency domain, such that a flat spectrum
# with amplitude s**2 results in a time series with
# stddev s. The fourier array passed in here must
# already be normalized such that this holds. Compared
# to numpy's ffts, this means dividing the fourier
# array by sqrt(n).

def detvecs_old(fourier, srate, dets=None):
	ndet, nfreq = fourier.shape
	# Use bins with exponential spacing
	bins = enlib.bins.expbin(nfreq, nbin=100, nmin=10)
	nbin = bins.shape[0]
	# This may seem very arbitrary, but has shown itself to
	# work very well, and much better than the more advanced
	# mode selection I attempted.
	maxmodes = 3

	V, E = [], []
	Nd, Nu = np.zeros([nbin,ndet]), np.zeros([nbin,ndet])
	for bi, b in enumerate(bins):
		# What we actually measure
		bft = fourier[:,b[0]:b[1]]
		cov = measure_cov(bft)
		var = np.diag(cov)
		corr = cov/(var[:,None]*var[None,:])**0.5
		Nd[bi] = var
		# Dealing with the whole correlation matrix
		# is too expensive. We need a compressed version.
		# Keep eigenmodes that are significant, up to at most
		# maxmodes eigenmodes. Given N hits per cell in the covmat,
		# the corr will be wishart distributed
		# with stddev (1/N*(corr**2+1))**0.5
		e,v        = np.linalg.eigh(corr)
		# sort by eigenvalue
		inds = np.argsort(e)[::-1]
		e, v = e[inds], v[:,inds]
		corr_var   = (corr**2+1)/(b[1]-b[0])
		sigma_e    = np.sum(v*corr_var.dot(v),0)**0.5

		detectable = e > 3*sigma_e
		e,v = e[detectable], v[:,detectable]
		e,v = e[:maxmodes],  v[:,:maxmodes]

		# Rescale so that these apply to the covmat rather
		# than corr matrix, to save some computation
		# when using the model.
		vcov = v*var[:,None]**0.5
		norm = np.sum(vcov**2,0)
		vcov/= norm[None,:]**0.5
		ecov = e*norm
		# We will represent the noise as (Nu + Nc)
		# (uncorrelated noise + correlated noise)
		# Nc is represented by V and E. Nu is what
		# extra variance is left in Nd. We impose a maximum
		# correlation of 0.9999 by adding a bit more noise.
		# This will make us downweigh the low frequencies
		# slightly too much, but avoids rounding errors
		# and unrealistically low levels of uncorrelated
		# noise.
		Nu[bi] = Nd[bi] - np.minimum(np.sum(vcov**2*ecov[None,:],1), Nd[bi]*0.9999)

		V.append(vcov)
		E.append(ecov)

	return prepare_detvecs(Nu, V, E, bins, srate, dets)

def detvecs_simple(fourier, srate, dets=None):
	nfreq = fourier.shape[1]
	ndet  = fourier.shape[0]

	bins_power = utils.expbin(nfreq, nbin=100, nmin=10)
	nbin  = bins_power.shape[0] # expbin may not provide exactly what we want
	Nd = np.empty((nbin,ndet))

	for bi, b in enumerate(bins_power):
		d     = fourier[:,b[0]:b[1]]
		Nd[bi] = measure_power(d)
	V = np.zeros([nbin,0,ndet])
	E = np.zeros([nbin,0])
	return prepare_detvecs(Nd, V, E, bins_power, srate, dets)

def detvecs_jon(ft, srate, dets=None, shared=False):
	nfreq  = ft.shape[1]
	# Construct our frequency bins
	bins = makebins([
			0.10, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 4.00,
			4.80, 5.60, 6.30, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 35.0, 38.0, 41.0,
			45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		], srate, nfreq)
	# Construct our mode bins. Interestingly, we skip
	# the f < 0.25 Hz area.
	mbins = makebins([0.25, 4.0], srate, nfreq)[1:]
	amp_thresholds = extend_list([6**2,5**2], len(mbins))
	single_threshold = 0.55
	white_scale = extend_list([1e-4, 0.25, 0.50, 1.00], len(bins))

	# Ok, compute our modes, and then measure them in each bin
	vecs = find_modes_jon(ft, mbins, amp_thresholds, single_threshold)
	assert vecs.size > 0, "Could not find any noise modes!"
	E, V, Nu, Nd = [], [], [], []
	vinds = []
	for bi, b in enumerate(bins):
		nmax = 500
		d    = ft[:,b[0]:b[1]:max(1,(b[1]-b[0])/nmax)]
		amps = vecs.T.dot(d)
		E.append(np.mean(np.abs(amps)**2,1))
		# Project out modes for every frequency individually
		dclean = d - vecs.dot(amps)
		# The rest is assumed to be uncorrelated
		Nu.append(np.mean(np.abs(dclean)**2,1)/white_scale[bi])
		Nd.append(np.mean(np.abs(d)**2,1))
		V.append(vecs)
		vinds.append(0)
	if shared:
		res = prepare_sharedvecs(Nu, V[:1], E, bins, srate, dets, vinds)
	else:
		res = prepare_detvecs(Nu, V, E, bins, srate, dets)
	return res


def prepare_detvecs(D, Vlist, Elist, ibins, srate, dets):
	D = np.asarray(D)
	if dets is None: dets = np.arange(D.shape[1])
	assert len(dets) == D.shape[1]
	fbins = ibins*(srate/2)/ibins[-1,-1]
	etmp = np.concatenate([[0],np.cumsum(np.array([len(e) for e in Elist]))])
	ebins= np.array([etmp[0:-1],etmp[1:]]).T
	E, V = np.hstack(Elist), np.hstack(Vlist).T
	return nmat.NmatDetvecs(D, V, E, fbins, ebins, dets)

def prepare_sharedvecs(D, Vlist, Elist, ibins, srate, dets, vinds):
	D = np.asarray(D)
	if dets is None: dets = np.arange(D.shape[1])
	assert len(dets) == D.shape[1]
	fbins = ibins*(srate/2)/ibins[-1,-1]
	etmp = np.concatenate([[0],np.cumsum(np.array([len(e) for e in Elist]))])
	ebins= np.array([etmp[0:-1],etmp[1:]]).T
	vtmp = np.concatenate([[0],np.cumsum(np.array([len(v) for v in Vlist]))])
	vbins= np.array([vtmp[i:i+2] for i in vinds])
	E, V = np.hstack(Elist), np.hstack(Vlist).T
	return nmat.NmatSharedvecs(D, V, E, fbins, ebins, vbins, dets)

def measure_cov(d, nmax=5000):
	d = d[:,::max(1,d.shape[1]/nmax)]
	(n,m) = d.shape
	step  = 1000
	res = np.zeros((n,n))
	for i in range(0,m,step):
		sub = d[:,i:i+step]
		res += np.real(sub.dot(np.conj(sub.T)))
	return res/m
def project_out(d, modes): return d-modes.T.dot(modes.dot(d))
def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def makebins(edge_freqs, srate, nfreq):
	binds  = (np.asarray(edge_freqs)/(srate/2.0)*nfreq).astype(int)
	return np.array([np.concatenate([[0],binds]),np.concatenate([binds,[nfreq]])]).T

def project_out_from_matrix(A, V):
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def find_modes_jon(ft, bins, amp_thresholds=None, single_threshold=0):
	ndet = ft.shape[0]
	vecs = np.zeros([ndet,0])
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		cov  = measure_cov(d)
		cov  = project_out_from_matrix(cov, vecs)
		e, v = np.linalg.eigh(cov)
		if amp_thresholds != None:
			good = e > amp_thresholds[bi]*np.median(e)
			e, v = e[good], v[:,good]
		if single_threshold and e.size:
			good = np.max(np.abs(v),0)<single_threshold
			e, v = e[good], v[:,good]
		vecs = np.hstack([vecs,v])
	return vecs

def extend_list(a, n): return a + [a[-1]]*(n-len(a))
