import numpy as np, scipy as sp, time, h5py
from enlib import nmat, utils, array_ops, fft, errors, config, gapfill

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
config.default("nmat_jon_apod", 0, "Apodization factor to apply for Jon's noise model")
config.default("nmat_jon_downweight", True, "Whether to downweight the lowest frequencies in the noise model.")

def detvecs_jon(ft, srate, dets=None, shared=False, cut_bins=None, apodization=None):
	"""Build a Detvecs noise matrix based on Jon's noise model.
	ft is the *normalized* fourier-transform of a TOD: ft = fft.rfft(d)/nsamp.
	srate is the sampling rate, dets is the list of detectors, shared specifies
	whether the Detvecs object should use the compressed "shared" layout or not",
	and cut_bins is a [nbin,{freq_from,freq_2}] array of frequencies
	to completely cut."""
	apodization = config.get("nmat_jon_apod", apodization) or None
	downweight  = config.get("nmat_jon_downweight")
	nfreq    = ft.shape[1]
	cut_bins = freq2ind(cut_bins, srate, nfreq)
	mask     = bins2mask(cut_bins, nfreq)
	# Construct our mode bins. Interestingly, we skip
	# the f < 0.25 Hz area.
	mbins = makebins([0.25, 4.0], srate, nfreq, 1000)[1:]
	amp_thresholds = extend_list([6**2,5**2], len(mbins))
	single_threshold = 0.55
	# Ok, compute our modes, and then measure them in each bin.
	# When using apodization, the vecs are not necessarily orthogonal,
	# so don't rely on that.
	vecs, weights = find_modes_jon(ft, mbins, amp_thresholds, single_threshold, mask=mask, apodization=apodization)
	bin_edges = np.array([
			0.10, 0.25, 0.35, 0.45, 0.55, 0.70, 0.85, 1.00,
			1.20, 1.40, 1.70, 2.00, 2.40, 2.80, 3.40, 4.00,
			4.80, 5.60, 6.30, 7.00, 8.00, 9.00, 10.0, 11.0,
			12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0,
			24.0, 26.0, 28.0, 30.0, 32.0, 35.0, 38.0, 41.0,
			45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 80.0, 90.0,
			100., 110., 120., 130., 140., 150., 160., 170.,
			180., 190.
		])
	# Cut bins that extend beyond our max frequency
	bin_edges = bin_edges[bin_edges < srate/2 * 0.99]
	bins = makebins(bin_edges, srate, nfreq, 2*vecs.shape[1], rfun=np.round)

	if downweight: white_scale = extend_list([1e-4, 0.25, 0.50, 1.00], len(bins))
	else: white_scale = [1]*len(bins)
	if vecs.size == 0: raise errors.ModelError("Could not find any noise modes")
	# Sharedvecs supports different sets of vecs per bin. But we only
	# use a single group here. So every bin refers to the first group.
	V     = [vecs]
	vinds = np.zeros(len(bins),dtype=int)
	Nu, Nd, E = measure_detvecs_bin(ft, bins, vecs, mask, weights)
	# Apply white noise scaling
	Nu /= np.array(white_scale)[:,None]
	if cut_bins is not None:
		bins, E, V, Nu, vinds = apply_bin_cuts(bins, cut_bins, E, V, Nu, vinds)
	if shared:
		res = prepare_sharedvecs(Nu, V, E, bins, srate, dets, vinds)
	else:
		# Expand V so we have one block of vectors per bin
		V = [V[i] for i in vinds]
		res = prepare_detvecs(Nu, V, E, bins, srate, dets)
	return res

def measure_detvecs_bin(ft, bins, vecs, mask, weights=None):
	Nu, Nd, E = [], [], []
	for bi, b in enumerate(bins):
		b = np.maximum(1,b)
		# Set up modes to use
		dm = mask[b[0]:b[1]]
		d  = ft[:,b[0]:b[1]]
		# Apply mask unless it would mask all modes
		if np.any(dm): d = d[:,dm]
		# Measure amps when we have non-orthogonal vecs
		rhs  = vecs.T.dot(d)
		div  = vecs.T.dot(vecs)
		amps = np.linalg.solve(div,rhs)
		if weights is not None:
			# Apply weights from apodization. This makes the separation between
			# what's modelled as uncorrelated and what's modelled as correlated
			# modes less abrupt. Since both white noise cleaning and E are measured
			# from amps, this is the only place that needs to care about the apodization.
			amps *= weights[:,None]**0.5
		E.append(np.mean(np.abs(amps)**2,1))
		# Project out modes for every frequency individually
		dclean = d - vecs.dot(amps)
		# The rest is assumed to be uncorrelated
		Nu.append(np.mean(np.abs(dclean)**2,1))
		Nd.append(np.mean(np.abs(d)**2,1))
	Nu = np.asarray(Nu)
	Nd = np.asarray(Nd)
	E  = np.asarray(E)
	return Nu, Nd, E

def calc_mean_ps(ft, chunk_size=32):
	# We do this in bunches to save memory
	res = np.zeros(ft.shape[-1],ft.real.dtype)
	for i in range(0, ft.shape[0], chunk_size):
		res += np.sum(np.abs(ft[i:i+chunk_size])**2,0)
	res /= ft.shape[0]
	return res

def detvecs_scaled(ft, srate, dets=None):
	# First set up our sample points
	mps  = calc_mean_ps(ft)
	bins = build_spec_bins(mps, ft.shape[0])
	freqs = utils.bins2edges(bins)
	# Build a linear model for the variance
	# FIXME: Not implemented yet
	vars  = fit_linear_coeffs(np.abs(ft)**2, freqs)
	# Scale the ft to compensate
	ft_scaled = ft.copy()
	core = nmat.get_core(ft.dtype)
	ifreqs = nmat.get_ifreqs(freqs, (ft.shape[-1]-1)*2)
	core.scale_lin(ft_scaled.T, ifreqs, vars.T, -0.5)
	# Build interior noise model
	nmat_inner = detvecs_jon(ft_scaled, dets=dets)
	return nmat.NmatScaled2(vars, freqs, nmat_inner)

config.default("nmat_uncorr_nbin",   100, "Number of bins for uncorrelated noise matrix")
config.default("nmat_uncorr_type", "exp", "Bin profile for uncorrelated noise matrix")
config.default("nmat_uncorr_nmin",    10, "Min modes per bin in uncorrelated noise matrix")
def detvecs_simple(fourier, srate, dets=None, type=None, nbin=None, nmin=None, vecs=None, eigs=None):
	nfreq = fourier.shape[1]
	ndet  = fourier.shape[0]
	type  = config.get("nmat_uncorr_type", type)
	nbin  = config.get("nmat_uncorr_nbin", nbin)
	nmin  = config.get("nmat_uncorr_nmin", nmin)

	if type is "exp":
		bins = utils.expbin(nfreq, nbin=nbin, nmin=nmin)
	elif type is "lin":
		bins = utils.linbin(nfreq, nbin=nbin, nmin=nmin)
	else: raise ValueError("No such power binning type '%s'" % type)
	nbin  = bins.shape[0] # expbin may not provide exactly what we want

	if vecs is None: vecs = np.full([ndet,0],1)
	# Initialize our noise vectors with default values
	vecs = np.asarray(vecs)
	nvec = vecs.shape[-1]
	Nu    = np.zeros([nbin,ndet])
	E     = np.full([nbin,nvec],1e-10)
	V     = [vecs]
	vinds = np.zeros(nbin,dtype=int)
	for bi, b in enumerate(bins):
		d     = fourier[:,b[0]:b[1]]
		if vecs.size > 0:
			# Measure amps when we have non-orthogonal vecs
			rhs  = vecs.T.dot(d)
			div  = vecs.T.dot(vecs)
			amps = np.linalg.solve(div,rhs)
			E[bi] = np.mean(np.abs(amps)**2,1)
			# Project out modes for every frequency individually
			d -= vecs.dot(amps)
		Nu[bi] = measure_power(d)
	# Override eigenvalues if necessary. This is useful
	# for e.g. forcing total common mode subtraction.
	# eigs must be broadcastable to [nbin,ndet]
	if eigs is not None: E[:] = eigs
	#return prepare_detvecs(Nd, V, E, bins, srate, dets)
	return prepare_sharedvecs(Nu, V, E, bins, srate, dets, vinds)

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
	# Used Woodbury to project out the given vectors from the
	# covmat A
	if V.size == 0: return A
	Q = A.dot(V)
	return A - Q.dot(np.linalg.solve(np.conj(V.T).dot(Q), np.conj(Q.T)))

def project_out_from_matrix_weighted(A, V, W):
	# Like project_out_from_matrix, but scales modes
	# by W**0.5 before subtracting them from A. This means that
	# they are partially left in A, with eigenvalues of (1-W)E.
	if V.size == 0: return A
	AV  = A.dot(V)
	AVW = A.dot(V*W[None]**0.5)
	return A - AVW.dot(np.linalg.solve(np.conj(V.T).dot(AV), np.conj(AVW.T)))

def find_modes_jon(ft, bins, amp_thresholds=None, single_threshold=0, mask=None, skip_mean=False, apodization=10, apod_threshold=0.02):
	if mask is None: mask = np.full(ft.shape[1], True, dtype=bool)
	if apodization is None:
		apodization = np.inf
		apod_threshold = 1.0
	ndet = ft.shape[0]
	vecs = np.zeros([ndet,0])
	if not skip_mean:
		# Force the uniform common mode to be included. This
		# assumes all the detectors have accurately measured gain.
		# Forcing this avoids the possibility that we don't find
		# any modes at all.
		vecs = np.hstack([vecs,np.full([ndet,1],ndet**-0.5)])
	scores = np.full(vecs.shape[1],1.0)
	for bi, b in enumerate(bins):
		d    = ft[:,b[0]:b[1]]
		# Ignore frequences in mask
		dm   = mask[b[0]:b[1]]
		if np.any(dm): d = d[:,dm]
		cov  = array_ops.measure_cov(d)
		#cov  = project_out_from_matrix(cov, vecs)
		cov  = project_out_from_matrix_weighted(cov, vecs, scores)
		# Should use eigh here. Something strange is going on on my
		# laptop here. Symptoms fit memory corruption. Writing out
		# cov and reading it in in a separate program and using eigh
		# there works. But calling it here results in noe eigenvalues
		# being nan. Should investigate.
		e, v = np.linalg.eig(cov)
		e, v = e.real, v.real
		score = np.full(len(e),1.0)
		if amp_thresholds != None:
			# Compute median, exempting modes we don't have enough
			# data to measure
			median_e = np.median(np.sort(e)[::-1][:b[1]-b[0]+1])
			score *= np.minimum(1,np.maximum(0,e/(amp_thresholds[bi]*median_e)))**apodization
		if single_threshold and e.size:
			# Reject modes too concentrated into a single mode. Judge based on
			# 1-fraction_in_single to make apodization smoother
			distributedness = 1-np.max(np.abs(v),0)
			score *= np.minimum(1,distributedness/(1-single_threshold))**apodization
		good = score >= apod_threshold
		e, v, score = e[good], v[:,good], score[good]
		vecs = np.hstack([vecs,v])
		scores = np.concatenate([scores,score])
	return vecs, scores

def extend_list(a, n): return a + [a[-1]]*(n-len(a))

class NmatBuildDelayed(nmat.NoiseMatrix):
	def __init__(self, model="jon", spikes=None, cut=None):
		self.model  = model
		self.spikes = spikes
		self.cut    = cut
	def update(self, tod, srate):
		# If we have noise estimation cuts, we must gapfill these
		# before measuring the noise, and restore them afterwards
		if self.cut is not None:
			vals = self.cut.extract(tod)
			gapfill.gapfill(tod, self.cut, inplace=True)
		try:
			if self.model == "jon":
				ft = fft.rfft(tod) * tod.shape[1]**-0.5
				noise_model = detvecs_jon(ft, srate, cut_bins=self.spikes)
			elif self.model == "uncorr":
				ft = fft.rfft(tod) * tod.shape[1]**-0.5
				noise_model = detvecs_simple(ft, srate)
			elif self.model == "white":
				noise_model = nmat.NoiseMatrix()
			else:
				raise ValueError("Unknown noise model '%s'" % self.model)
		except (errors.ModelError, np.linalg.LinAlgError, AssertionError) as e:
			print "Warning: Noise model fit failed for tod with shape %s. Assigning zero weight" % str(tod.shape)
			noise_model = nmat.NmatNull()
		if self.cut is not None:
			self.cut.insert(tod, vals)
		return noise_model
	def __getitem__(self, sel):
		res, detslice, sampslice = self.getitem_helper(sel)
		res.cut = res.cut[sel]
		return res

def build_spec_bins(mps, ndof, lim=5, bsize_density=40):
	ndof  = np.full(mps.shape, ndof*1.0)
	rbins = build_spec_bins(mps, ndof, lim=lim)
	# Areas with unusual statistics may end up with unrealistically
	# dense bins. We increase the bin threshold in these regions, and
	# recompute. This is ugly.
	bdens = calc_bin_density(rbins, bsize=bsize_density)
	ndof[:] = ndet/np.maximum(1, bdens/2)
	pbins = build_spec_bins(mps, ndof, lim=lim)
	return pbins

def build_spec_bins_helper(mps, ndof, lim=5, level=0):
	"""Build a noise spectrum binning for spectrum
	mps[nfreq] where each value is approximately
	scaled-chisq-distributed with ndof degrees of freedom.
	Bins are merged until they differ by less than lim sigma."""
	# The relative uncertainty of a chisq with ndof is
	# dev/mean = sqrt(2k)/k = sqrt(2/k)
	mvar  = mps**2*2/ndof
	# Let's use an iterative merging technique:
	# Start with full resolution diffs. Find values more
	# than lim significant. Make these bin boundaries.
	# For each area between the bin boundaries, downsample
	# by a factor and repeat the algorithm on these. Can
	# implement this looping in fortran if need be.
	diffs = mps[1:]-mps[:-1]
	dvars = mvar[1:]+mvar[:-1]
	if level == 0:
		with h5py.File("tmp.hdf","w") as hfile:
			hfile["data"] = np.abs(diffs)/dvars**0.5
	edges = np.where(np.abs(diffs)/dvars**0.5 > lim)[0]
	edges = np.concatenate([[0],edges+1,[mps.size]])
	ibins = np.array([edges[:-1],edges[1:]]).T
	obins = []
	for b in ibins:
		bsize = b[1]-b[0]
		if bsize > 10:
			# Downsample data in bin and recurse
			bmps  = downsample(mps[b[0]:b[1]],  2)
			bndof = downsample(ndof[b[0]:b[1]], 2)*2
			obins.append(b[0]+2*build_spec_bins(bmps, bndof, lim=lim, level=level+1))
		else:
			obins.append([b])
	obins = np.concatenate(obins)
	return obins

def calc_bin_density(bins, bsize=40):
	edges  = np.unique(np.concatenate([np.arange(0,bins[-1,1],bsize),[bins[-1,1]]]))
	obins  = utils.edges2bins(edges)
	counts = np.zeros(len(obins))
	for bi, b in enumerate(bins):
		i1,i2 = b/bsize
		counts[b[0]/bsize:b[1]/bsize+1] += 1
	return utils.bin_expand(obins, counts)

def downsample(a, n):
	nb  = a.size/n
	res = np.zeros((a.size+n-1)/n)
	res[:nb] = np.mean(a[:nb*n].reshape(nb,n),-1)
	if nb < res.size:
		res[-1] = np.mean(a[nb*n:])
	return res

