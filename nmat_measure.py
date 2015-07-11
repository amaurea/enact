import numpy as np, scipy as sp, enlib.bins, time, enlib.bins, h5py
from enlib import nmat, utils,array_ops

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

# Our main noise model
def detvecs_jon(ft, srate, dets=None, shared=False, cut_bins=None):
	"""Build a Detvecs noise matrix based on Jon's noise model.
	ft is the *normalized* fourier-transform of a TOD: ft = fft.rfft(d)/nsamp.
	srate is the sampling rate, dets is the list of detectors, shared specifies
	whether the Detvecs object should use the compressed "shared" layout or not",
	and cut_bins is a [nbin,{freq_from,freq_2}] array of frequencies
	to completely cut."""
	nfreq    = ft.shape[1]
	cut_bins = freq2ind(cut_bins, srate, nfreq)
	mask     = bins2mask(cut_bins, nfreq)
	# Construct our mode bins. Interestingly, we skip
	# the f < 0.25 Hz area.
	mbins = makebins([0.25, 4.0], srate, nfreq, 1000)[1:]
	amp_thresholds = extend_list([6**2,5**2], len(mbins))
	single_threshold = 0.55
	# Ok, compute our modes, and then measure them in each bin
	vecs = find_modes_jon(ft, mbins, amp_thresholds, single_threshold, mask=mask)
	bins = makebins([
			0.10, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 4.00,
			4.80, 5.60, 6.30, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 35.0, 38.0, 41.0,
			45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		], srate, nfreq, 2*vecs.shape[1], rfun=np.round)

	white_scale = extend_list([1e-4, 0.25, 0.50, 1.00], len(bins))
	assert vecs.size > 0, "Could not find any noise modes!"
	E, V, Nu, Nd = [], [vecs], [], []
	vinds = []
	for bi, b in enumerate(bins):
		nmax = 1000
		b = np.maximum(1,b)
		# Set up modes to use
		dm = mask[b[0]:b[1]]
		d  = ft[:,b[0]:b[1]]
		# Apply mask unless it would mask all modes
		if np.any(dm): d = d[:,dm]
		# Save time by only using a subset of samples for estimating correlations
		#d = sample_nmax(d,nmax)
		amps = vecs.T.dot(d)
		E.append(np.mean(np.abs(amps)**2,1))
		# Project out modes for every frequency individually
		dclean = d - vecs.dot(amps)
		# The rest is assumed to be uncorrelated
		Nu.append(np.mean(np.abs(dclean)**2,1)/white_scale[bi])
		Nd.append(np.mean(np.abs(d)**2,1))
		vinds.append(0)
	if cut_bins is not None:
		bins, E, V, Nu, vinds = apply_bin_cuts(bins, cut_bins, E, V, Nu, vinds)
	if shared:
		res = prepare_sharedvecs(Nu, V, E, bins, srate, dets, vinds)
	else:
		# Expand V so we have one block of vectors per bin
		V = [V[i] for i in vinds]
		res = prepare_detvecs(Nu, V, E, bins, srate, dets)
	return res

def detvecs_simple(fourier, srate, dets=None):
	nfreq = fourier.shape[1]
	ndet  = fourier.shape[0]

	bins_power = enlib.bins.expbin(nfreq, nbin=100, nmin=10)
	nbin  = bins_power.shape[0] # expbin may not provide exactly what we want
	Nd = np.empty((nbin,ndet))

	for bi, b in enumerate(bins_power):
		d     = fourier[:,b[0]:b[1]]
		Nd[bi] = measure_power(d)
	V = [np.full([ndet,0],1)]*nbin
	E = [np.full([0],1e-10)]*nbin
	return prepare_detvecs(Nd, V, E, bins_power, srate, dets)

# This one should have been better than Jon's model, but simulations
# show that nmat=build(sqrt(C)r1); nmat.apply(Cr2) is whiter for his
# model than this one.
def detvecs_joint(ft, srate, dets=None, cut_bins=None, nbin=50, samp_min=75, samp_max=2000, maxmodes=15, mineig=0.005):
	ndet, nfreq = ft.shape
	cut_bins = freq2ind(cut_bins, srate, nfreq)
	mask     = bins2mask(cut_bins, nfreq)
	# First define our bins. We want at least samp_min and at most_samp_max samples
	# per bin. Too few, and we can't measure the covariance accurately. Too many, and we lose
	# resolution.
	bins = enlib.bins.expbin(nfreq, nbin, samp_min, samp_max)
	# For each bin, measure the covariance and decompe it into detvecs
	params = []
	for bi, b in enumerate(bins):
		dm  = mask[b[0]:b[1]]
		# Mask cut bins unless that would remove all samples in this bin
		d = ft[:,b[0]:b[1]]
		if np.any(dm) > 1: d = d[:,dm]
		cov = measure_cov(d)
		params.append(nmat.decomp_DVEV(cov, nmax=min(maxmodes,d.shape[1]/20), mineig=mineig))
	D,E,V = map(list, zip(*params))
	# Split into cut and uncut regions
	if cut_bins is not None:
		bins, E, V, D = apply_bin_cuts(bins, cut_bins, E, V, D)
	return prepare_detvecs(D, V, E, bins, srate, dets)

def detvecs_bigjoint(ft, srate, dets=None, cut_bins=None, nbin=50, samp_min=50, samp_max=1000, maxmodes=30, mineig=0.001):
	ndet, nfreq = ft.shape
	cut_bins = freq2ind(cut_bins, srate, nfreq)
	mask     = bins2mask(cut_bins, nfreq)
	# First define our bins. We want at least samp_min and at most_samp_max samples
	# per bin. Too few, and we can't measure the covariance accurately. Too many, and we lose
	# resolution.
	bins = enlib.bins.expbin(nfreq, nbin, samp_min, samp_max)
	# First measure the overall correlation pattern of the whole thing. We measure corr per bin,
	# and then take the mean corr. This will avoid giving undue weight to the atmosphere, which
	# would otherwise swamp everything
	C = np.zeros([ndet,ndet])
	covs = []
	for b in bins:
		dm  = mask[b[0]:b[1]]
		# Mask cut bins unless that would remove all samples in this bin
		d = ft[:,b[0]:b[1]]
		if np.any(dm) > 1: d = d[:,dm]
		cov = measure_cov(ft[:,b[0]:b[1]])
		C  += cov/np.mean(np.diag(cov))
		covs.append(cov)
	# Extract eigenmodes from this (scaled) correlation matrix
	_,_,V = nmat.decomp_DVEV(C, nmax=maxmodes, mineig=mineig)
	# Use the highest bin variance as a proxy for the uncorrelated noise
	Dapprox = np.diag(covs[-1])
	# Then keep V fixed and fit C=D+VEV' for each bin
	params = []
	for bi, (b,C) in enumerate(zip(bins,covs)):
		dm  = mask[b[0]:b[1]]
		d = ft[:,b[0]:b[1]]
		if np.any(dm) > 1: d = d[:,dm]
		D,E = decomp_clean_modes(d, V, Dapprox)
		params.append((D,E,V))
	D,E,V = map(list, zip(*params))
	# Split into cut and uncut regions
	if cut_bins is not None:
		bins, E, V, D = apply_bin_cuts(bins, cut_bins, E, V, D)
	return prepare_detvecs(D, V, E, bins, srate, dets)

def decomp_clean_modes(d, V, D):
	def getamps(W, V, d):
		rhs = (V.T*W[None]).dot(d)
		A   = (V.T*W[None]).dot(V)
		iA  = array_ops.eigpow(A,-1)
		return iA.dot(rhs)
		#return np.linalg.solve(A,rhs)
	# Solve for the amplitudes of the modes in V per mode
	#e = getamps(1/D, V, d)
	e = V.T.dot(d)
	# Remove from time-stream
	d2= d-V.dot(e)
	# Estimate D from the cleaned stream
	D = np.var(d2,1)
	# E is the average value of e
	E = np.mean(np.abs(e)**2,1)
	return D, E

def apply_bin_cuts(bins, cut_bins, E, V, Nu, vinds=None):
	# Insert cuts into bins, possibly splitting them.
	# Will insert uncorrelated infinite noise bins at cut locations.
	ndet  = Nu[0].size
	Vcut  = np.zeros([ndet,0])
	Ecut  = np.zeros([0])
	Nucut = np.full([ndet],np.max(Nu)*1e6)
	bsplit, rmap, abmap = utils.range_sub(bins, cut_bins, mapping=True)
	E2, Nu2, bins2 = [],[],[]
	# Handle non-correlated part
	for i in abmap:
		if i < 0:
			# In cut range
			E2.append(Ecut)
			Nu2.append(Nucut)
			bins2.append(cut_bins[-i-1])
		else:
			old_ind = rmap[i]
			E2.append(E[old_ind])
			Nu2.append(Nu[old_ind])
			bins2.append(bsplit[i])
	bins2 = np.array(bins2)
	# Correlated part is different for shared and unshared vectors
	if vinds is None:
		V2 = [Vcut if i < 0 else V[rmap[i]] for i in abmap]
		return bins2, E2, V2, Nu2
	else:
		V2 = V+[Vcut]
		vinds2 = [len(V) if i < 0 else vinds[rmap[i]] for i in abmap]
		return bins2, E2, V2, Nu2, vinds2

def prepare_detvecs(D, Vlist, Elist, ibins, srate, dets):
	D = np.asarray(D)
	if dets is None: dets = np.arange(D.shape[1])
	assert len(dets) == D.shape[1]
	fbins = ibins*(srate/2.)/ibins[-1,-1]
	etmp = np.concatenate([[0],np.cumsum(np.array([len(e) for e in Elist]))])
	ebins= np.array([etmp[0:-1],etmp[1:]]).T
	E, V = np.hstack(Elist), np.hstack(Vlist).T
	return nmat.NmatDetvecs(D, V, E, fbins, ebins, dets)

def prepare_sharedvecs(D, Vlist, Elist, ibins, srate, dets, vinds):
	"""Construct an NmatSharedvecs based on uncorrelated noise D[nbin,ndet],
	correlated modes V[ngroup][nmode,ndet], mode amplitudes E[nbin][nmode],
	integer bin start/stops ibins[nbin,2], sampling rate srate, detectors
	dets[ndet] and an index for which mode group applies to which bin,
	vinds[nbin]"""
	D = np.asarray(D)
	if dets is None: dets = np.arange(D.shape[1])
	assert len(dets) == D.shape[1]
	fbins = ibins*(srate/2.)/ibins[-1,-1]
	etmp = np.concatenate([[0],np.cumsum(np.array([len(e) for e in Elist]))])
	ebins= np.array([etmp[0:-1],etmp[1:]]).T
	vtmp = np.concatenate([[0],np.cumsum(np.array([len(v.T) for v in Vlist]))])
	vbins= np.array([vtmp[i:i+2] for i in vinds])
	E, V = np.hstack(Elist), np.hstack(Vlist).T
	return nmat.NmatSharedvecs(D, V, E, fbins, ebins, vbins, dets)

def mycontiguous(a):
	b = np.zeros(a.shape, a.dtype)
	b[...] = a[...]
	return b

def measure_cov(d, nmax=10000):
	d = d[:,::max(1,d.shape[1]/nmax)]
	n,m = d.shape
	step  = 10000
	res = np.zeros((n,n))
	for i in range(0,m,step):
		sub = mycontiguous(d[:,i:i+step])
		res += np.real(sub.dot(np.conj(sub.T)))
	return res/m
def project_out(d, modes): return d-modes.T.dot(modes.dot(d))
def measure_power(d): return np.real(np.mean(d*np.conj(d),-1))

def freq2ind(freqs, srate, nfreq, rfun=None):
	"""Returns the index of the first fourier mode with greater than freq
	frequency, for each freq in freqs."""
	if freqs is None: return freqs
	if rfun  is None: rfun = np.ceil
	return rfun(np.asarray(freqs)/(srate/2.0)*nfreq).astype(int)

def makebins(edge_freqs, srate, nfreq, nmin=0, rfun=None):
	binds  = freq2ind(edge_freqs, srate, nfreq, rfun=rfun)
	if nmin > 0:
		binds2 = [binds[0]]
		for b in binds:
			if b-binds2[-1] >= nmin: binds2.append(b)
		binds = binds2
	return np.array([np.concatenate([[0],binds]),np.concatenate([binds,[nfreq]])]).T

def sample_nmax(d, nmax):
	step = max(d.shape[-1]/nmax,1)
	return d[:,::step]

def bins2mask(bins, nfreq):
	mask = np.full(nfreq, True, dtype=bool)
	if bins is not None:
		for b in bins: mask[b[0]:b[1]] = False
	return mask

def project_out_from_matrix(A, V):
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def find_modes_jon(ft, bins, amp_thresholds=None, single_threshold=0, mask=None):
	if mask is None: mask = np.full(ft.shape[1], True, dtype=bool)
	ndet = ft.shape[0]
	vecs = np.zeros([ndet,0])
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		# Ignore frequences in mask
		dm   = mask[b[0]:b[1]]
		if np.any(dm): d = d[:,dm]
		cov  = array_ops.measure_cov(d)
		cov  = project_out_from_matrix(cov, vecs)
		# Should use eigh here. Something strange is going on on my
		# laptop here. Symptoms fit memory corruption. Writing out
		# cov and reading it in in a separate program and using eigh
		# there works. But calling it here results in noe eigenvalues
		# being nan. Should investigate.
		e, v = np.linalg.eig(cov)
		e = e.real
		v = v.real
		if amp_thresholds != None:
			good = e > amp_thresholds[bi]*np.median(e)
			e, v = e[good], v[:,good]
		if single_threshold and e.size:
			good = np.max(np.abs(v),0)<single_threshold
			e, v = e[good], v[:,good]
		vecs = np.hstack([vecs,v])
	return vecs

def extend_list(a, n): return a + [a[-1]]*(n-len(a))
