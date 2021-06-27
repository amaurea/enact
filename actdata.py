from __future__ import division, print_function
import numpy as np, time, os, multiprocessing, sys
from scipy import signal
from enlib import utils, dataset, nmat, config, errors, gapfill, fft, pointsrcs, todops, bunch, bench, sampcut, coordinates
from enact import files, cuts, filters
from numpy.lib import recfunctions

try: basestring
except: basestring = str

def expand_file_params(params, top=True):
	"""In general we expect parameters to be given as dictionaries
	with key-value pairs, where one is usually the file name. However,
	a very common case is for there just to be a file name, so we
	support just giving the file name and expand it into a dict here.
	Additionally, for backwards compatibility (and extra flexibility)
	we also support lists of such parameter dicts. This function
	always returns a list."""
	if isinstance(params, basestring):
		params = {"fname": params}
	elif isinstance(params, list) or isinstance(params, tuple):
		params = [expand_file_params(p, False) for p in params]
	if top and not isinstance(params, list):
		params = [params]
	return params

def get_array_name(tod_id): return tod_id.split(".")[-1].replace("ar","pa")

def build_detname(uids, entry=None, tag=None):
	if tag is None: tag = get_array_name(entry.id)
	return np.char.mod(tag + "_%04d", uids)
def split_detname(detnames):
	if len(detnames) == 0: return "", np.array([],int)
	toks = np.char.partition(detnames, "_")
	return toks[:,0], toks[:,2].astype(int)

def try_read(method, desc, params, *args, **kwargs):
	"""Try to read multiple alternative filenames, raising a DataMissing
	exception only if none of them can be read. Otherwise, return the first
	matching."""
	params = expand_file_params(params)
	for param in params:
		kwargs2 = kwargs.copy()
		kwargs2.update(param)
		del kwargs2["fname"]
		try: return method(param["fname"], *args, **kwargs2)
		except (IOError, OSError,errors.DataMissing) as e: pass
	raise errors.DataMissing(desc + ": " + ", ".join([str(param) for param in params]))

def try_read_perdet(method, desc, params, *args, **kwargs):
	"""Try to read multiple alternative filenames, raising a DataMissing
	exception only if none of them can be read. Otherwise, return the first
	matching."""
	params = expand_file_params(params)
	odets, ovals = None, None
	for param in params:
		kwargs2 = kwargs.copy()
		kwargs2.update(param)
		del kwargs2["fname"]
		try:
			idets, ivals = method(param["fname"], *args, **kwargs2)
			if odets is None:
				odets, ovals = idets, ivals
			else:
				new = utils.find(odets, idets, -1)<0
				odets = np.concatenate([odets,idets[new]])
				ovals = np.concatenate([ovals,ivals[new]])
		except (IOError, OSError,errors.DataMissing) as e: pass
	if odets is None:
		raise errors.DataMissing(desc + ": " + ", ".join([str(param) for param in params]))
	else:
		order = np.argsort(odets)
		odets, ovals = odets[order], ovals[order]
		return odets, ovals

def get_dict_wild(d, key, default=None):
	if key in d: return d[key]
	if '*' in d: return d['*']
	if default is not None: return default
	raise KeyError(key)
def get_dict_default(d, key, default):
	if key in d: return d[key]
	else: return default

def try_read_dict(method, desc, params, key, *args, **kwargs):
	"""Try to find a value in one of files provided, using the given read
	method, which must return a dictionary."""
	params = expand_file_params(params)
	for param in params:
		kwargs2 = kwargs.copy()
		kwargs2.update(param)
		del kwargs2["fname"]
		try:
			dict = method(param["fname"], *args, **kwargs2)
			return get_dict_wild(dict, key)
		except (IOError, OSError, KeyError) as e: pass
	raise errors.DataMissing(desc + ": " + ", ".join([str(param) for param in params]))

def read_gain(entry):
	dets, gain_raw = try_read(files.read_gain, "gain", entry.gain)
	try:
		corrs = try_read(files.read_gain_correction, "gain_correction", entry.gain_correction, id=entry.id)
		correction  = get_dict_wild(corrs, entry.id)
	except KeyError: raise errors.DataMissing("gain_correction id: " + entry.id)
	mask = np.isfinite(gain_raw)*(gain_raw != 0)
	dets, gain_raw = dets[mask], gain_raw[mask]
	# Get the gain mode, which tells us how to compute the total gain
	gain_mode = entry.gain_mode if "gain_mode" in entry else "direct"
	datafields = [
		dataset.DataField("gain_raw", gain_raw, dets=build_detname(dets, entry), det_index=0),
		dataset.DataField("gain_correction", correction),
		dataset.DataField("gain_mode", gain_mode),
		dataset.DataField("entry", entry)]
	# Read the flatfield if available
	if "flatfield" in entry:
		flat_dets, flat_gain = try_read(files.read_flatfield, "flatfield", entry.flatfield)
		datafields.append(dataset.DataField("flatfield", flat_gain, dets=build_detname(flat_dets, entry), det_index=0))
	if "gain_factor" in entry:
		datafields.append(dataset.DataField("gain_factor", float(entry.gain_factor)))
	return dataset.DataSet(datafields)

def calibrate_gain(data):
	"""Combine raw gains and gain corrections to form the final gain."""
	require(data, ["gain_raw","gain_correction","tag_defs","gain_mode","mce_gain"])
	gain = data.gain_raw.copy()
	applied = np.zeros(gain.shape,int)
	for tag_name in data.gain_correction:
		if tag_name == "*":
			gain *= data.gain_correction[tag_name]
			applied += 1
		elif tag_name not in data.tag_defs:
			raise errors.DataMissing("Unrecognized tag in gain correction: '%s'" % tag_name)
		else:
			gain_inds, tag_inds = utils.common_inds([data.dets, data.tag_defs[tag_name]])
			gain[gain_inds] *= data.gain_correction[tag_name]
			applied[gain_inds] += 1
	uncorr   = np.where(applied<1)[0]
	overcorr = np.where(applied>1)[0]
	if len(uncorr) > 0:
		raise errors.DataMissing("Missing gain correction for dets [%s]" % ",".join([str(d) for d in uncorr]))
	if len(overcorr) > 0:
		raise errors.DataMissing("Multiple gain correction per detector for dets [%s]" % ",".join([str(d) for d in overcorr]))
	# Apply optional flatfield
	if "flatfield" in data:
		gain *= data.flatfield
	if "gain_factor" in data:
		gain *= data.gain_factor
	# Apply mce filter gain if necessary
	if data.gain_mode == "mce":
		gain /= float(data.mce_gain)
	elif data.gain_mode == "mce_compat":
		gain /= data.mce_gain/1217.8583043
	elif data.gain_mode == "direct": pass
	else: raise ValueError("Unrecognized gain_mode '%s'" % data.gain_mode)
	data += dataset.DataField("gain", gain, dets=data.dets, det_index=0)
	return data

def read_mce_filter(entry):
	params, f_samp = try_read(files.read_mce_filter_params, "mce_filter", entry.tod)
	K = 0.5**14
	b11,b12,b21,b22,k1,k2 = np.array(params)*[K,K,K,K,1,1]
	mce_gain = 2.**4/(1-b11+b12)/(1-b21+b22)/2**(k1+k2)
	data = dataset.DataSet([
		dataset.DataField("mce_fsamp",  f_samp),
		dataset.DataField("mce_gain",   mce_gain),
		dataset.DataField("mce_params", params)])
	return data

def read_polangle(entry):
	dets, data = try_read(files.read_polangle, "polangle", entry.polangle)
	return dataset.DataSet([
		dataset.DataField("polangle", data, dets=build_detname(dets, entry), det_index=0),
		dataset.DataField("entry", entry)])

def read_tconst(entry):
	dets, data = try_read(files.read_tconst, "tconst", entry.tconst, id=entry.id)
	return dataset.DataSet([
		dataset.DataField("tau", data, dets=build_detname(dets, entry), det_index=0),
		dataset.DataField("entry", entry)])

# There are 3 types of cuts:
# 1. Samples that should be cut when making maps, but not when estimating noise.
#    These are e.g. samples that hit moon sidelobes, or which have suspicious
#    statistical properties.
# 2. Samples that should be cut when estimating noise, but not when making maps.
#    For example samples that hit planets when mapping planets.
# 3. Samples that should be cut both when estimating noise and when making maps.
#    These are tyipcally glitches, planets, and other samples with huge values.
#
# All these can be handled using only two categories:
# 1. cut
# 2. cut_noise
# If cut_noise defaults to being equato to cut, then specifying only cut
# would recover the old behavior. cut can default to empty. Automatic
# cuts would add to cut.
#
# Each of these categories can be the union of several different cut sets,
# which can be in either the old or new format. Could have a single cut
# entry in the filedb, but that would make it hard to override only one
# of them

def merge_cuts(cutinfos):
	# each cutinfo is dets, cuts, offset. Find intersection of detectors
	detlists = [ci[0] for ci in cutinfos]
	dets     = utils.common_vals(detlists)
	detinds  = utils.common_inds(detlists)
	# Find the max offset and how much we must cut off the start of each member
	offsets  = np.array([ci[2] for ci in cutinfos])
	offset   = np.max(offsets)
	offrel   = offset - offsets
	# slice each cut
	cuts = [ci[1][d,o:] for ci,d,o in zip(cutinfos, detinds, offrel)]
	# And produce the cut sum
	cut  = cuts[0]
	for c in cuts[1:]: cut *= c
	return dets, cut, offset

def try_read_cut(params, desc, id):
	"""We support a more complicated format for cuts."""
	# If a list is given, try them one by one and use the first usable one
	if isinstance(params, list):
		messages = []
		for param in params:
			try:
				return try_read_cut(param, desc, id)
			except (IOError, OSError, errors.DataMissing) as e:
				messages.append(str(e))
		raise errors.DataMissing(desc + ": " + ", ".join([str(param) + ": " + mes for param,mes in zip(params, messages)]))
	# Convenience transformations, to make things a bit more readable in the parameter files
	if isinstance(params, basestring):
		toks = params.split(":")
		if toks[0].endswith(".hdf") or toks[0].endswith(".pdf") or toks[0].endswith(".h5"): params = {"type":"hdf","fname":toks[0],"flags":toks[1]}
		else: params = {"type":"old","fname":toks[0]}
	try:
		if   params["type"] == "old":
			permissive = True if "permissive" not in params else params["permissive"]
			return files.read_cut(params["fname"], permissive=permissive)
		elif params["type"] == "hdf": return files.read_cut_hdf(params["fname"], id=id, flags=params["flags"].split(","))
		elif params["type"] == "union":
			return merge_cuts([try_read_cut(param, desc, id) for param in params["subs"]])
		else: raise ValueError("Unrecognized cut type '%s'" % params["type"])
	except (IOError, OSError, errors.DataMissing) as e:
		raise errors.DataMissing(desc + ": " + str(e))

def read_cut(entry, names=["cut","cut_basic","cut_noiseest","cut_quality"], default="cut"):
	fields = [dataset.DataField("entry",entry)]
	for name in names:
		if name not in entry or entry[name] is None:
			if default not in entry or entry[default] is None:
				raise errors.DataMissing("Trying to read cut, but no cut data present!")
			param = entry[default]
		else: param = entry[name]
		dets, data, offset = try_read_cut(param, name, entry.id)
		samples = [offset, offset + data.nsamp]
		def stacker(cuts, axis): return sampcut.stack(cuts)
		fields.append(dataset.DataField(name, data, dets=build_detname(dets, entry), det_index=0, samples=samples, sample_index=1,
			stacker=stacker))
	return dataset.DataSet(fields)

def read_point_offsets(entry, no_correction=False):
	dets, template = try_read(files.read_point_template, "point_template", entry.point_template)
	correction = 0
	if not no_correction:
		correction = try_read_dict(files.read_point_offsets, "point_offsets", entry.point_offsets, entry.id)
	fields = [
		dataset.DataField("point_offset",  template+correction, dets=build_detname(dets, entry), det_index=0),
		dataset.DataField("point_template",template, dets=build_detname(dets, entry), det_index=0),
		dataset.DataField("point_correction",correction),
		dataset.DataField("entry", entry)]
	if not no_correction and "point_slopes" in entry:
		slope = try_read_dict(files.read_point_slopes, "point_slopes", entry.point_slopes, entry.id)
		fields.append(dataset.DataField("point_slope", slope))
	return dataset.DataSet(fields)

def read_site(entry):
	site = try_read(files.read_site, "site", entry.site)
	return dataset.DataSet([
		dataset.DataField("site", site),
		dataset.DataField("entry", entry)])

def read_noise(entry):
	data = try_read(nmat.read_nmat, "noise", entry.noise)
	return dataset.DataSet([
		dataset.DataField("noise", data, dets=build_detname(data.dets, entry), det_index=0),
		dataset.DataField("entry", entry)])

def read_beam(entry):
	beam = try_read(files.read_beam, "beam", entry.beam)
	return dataset.DataSet([
		dataset.DataField("beam", beam),
		dataset.DataField("entry", entry)])

def read_noise_cut(entry):
	try: dets = try_read(files.read_noise_cut, "noise_cut", entry.noise_cut, id=entry.id)[entry.id]
	except KeyError: raise errors.DataMissing("noise_cut id: " + entry.id)
	return dataset.DataSet([
		dataset.DataField("noise_cut", dets=build_detname(dets, entry)),
		dataset.DataField("entry", entry)])

def read_spikes(entry):
	spikes = try_read(files.read_spikes, "spikes", entry.spikes)
	return dataset.DataSet([
		dataset.DataField("spikes", data=spikes),
		dataset.DataField("entry", entry)])

def read_boresight(entry, moby=False):
	if moby: bore, flags = try_read(files.read_boresight_moby, "boresight", entry.tod)
	else:    bore, flags = try_read(files.read_boresight,      "boresight", entry.tod)
	return dataset.DataSet([
		dataset.DataField("boresight", bore, samples=[0,bore.shape[1]], sample_index=1, force_contiguous=True),
		dataset.DataField("flags",     flags,samples=[0,flags.shape[0]],sample_index=0),
		dataset.DataField("entry",     entry)])

def read_dark(entry):
	# Need to read array info to find out which detectors are dark
	ainfo = read_array_info(entry).array_info
	dark_dets = ainfo.info.det_uid[ainfo.info.det_type=="dark_tes"]
	# Then read the actual tod
	_, tod = try_read(files.read_tod, "dark_tod", entry.tod, ids=dark_dets)
	samples = [0,tod.shape[-1]]
	return dataset.DataSet([
		dataset.DataField("dark_dets", dark_dets),
		#dataset.DataField("dark_cut", cuts, samples=samples, sample_index=1),
		dataset.DataField("dark_tod", tod, samples=samples, sample_index=1)])

def read_buddies(entry):
	dets, buddies = try_read(files.read_buddies, "buddies", entry.buddies)
	return dataset.DataSet([
		dataset.DataField("buddies_raw", data=buddies),
		dataset.DataField("buddies_raw_dets", data=build_detname(dets, entry)),
	])

config.default("hwp_fallback", "none", "How to handle missing HWP data. 'none' skips the tod (it it is supposed to have hwp data), while 'raw' falls back on the native hwp data.")
def read_hwp(entry):
	dummy = dataset.DataSet([
		dataset.DataField("hwp", 0),
		dataset.DataField("hwp_id", "none"),
		dataset.DataField("hwp_source", "none")])
	epochs = try_read(files.read_hwp_epochs, "hwp_epochs", entry.hwp_epochs)
	t, _, ar = entry.id.split(".")
	t = float(t)
	if ar not in epochs: return dummy
	for epoch in epochs[ar]:
		if t >= epoch[0] and t < epoch[1]:
			# Ok, the HWP was active during this period. Try to read it. It can be in
			# several different formats.
			if   entry.hwp_format == "tod":
				# HWP angles in the tod, in the new, high-quality format
				hwp, flags = try_read(files.read_hwp_angle, "hwp_tod_angles", entry.tod)
				return dataset.DataSet([
					dataset.DataField("hwp", hwp, samples=[0, hwp.size], sample_index=0),
					dataset.DataField("hwp_id", epoch[2]),
					dataset.DataField("hwp_source", "tod")])
			elif entry.hwp_format == "raw":
				# HWP angles in the tod, in the old, inaccurate format
				hwp = try_read(files.read_hwp_raw, "hwp_raw_angles", entry.tod)
				return dataset.DataSet([
					dataset.DataField("hwp", hwp, samples=[0, hwp.size], sample_index=0),
					dataset.DataField("hwp_id", epoch[2]),
					dataset.DataField("hwp_source", "raw")])
			elif entry.hwp_format == "external":
				# HWP angles in external data files
				try:
					status = try_read(files.read_hwp_status, "hwp_status", entry.hwp_status)
				except errors.DataMissing as e:
					status = None
				# If there weren't any external files, possibly fall back to raw angles
				if status is None or get_dict_wild(status, entry.id, 0) != 1:
					if config.get("hwp_fallback") == "raw":
						hwp = try_read(files.read_hwp_raw, "hwp_raw_angles", entry.tod)
						return dataset.DataSet([
							dataset.DataField("hwp", hwp, samples=[0, hwp.size], sample_index=0),
							dataset.DataField("hwp_id", epoch[2]),
							dataset.DataField("hwp_source", "raw")])
					else:
						raise e if status is None else errors.DataMissing("Missing HWP angles!")
				# Try to read the angles themselves
				hwp = try_read(files.read_hwp_cleaned, "hwp_angles", entry.hwp)
				return dataset.DataSet([
					dataset.DataField("hwp", hwp, samples=[0,hwp.size], sample_index=0),
					dataset.DataField("hwp_id", epoch[2]),
					dataset.DataField("hwp_source","cleaned")])
	# Not in any epoch, so return 0 hwp angle (which effectively turns it off)
	return dummy

def read_layout(entry):
	data = try_read(files.read_layout, "layout", entry.layout)
	return dataset.DataSet([
		dataset.DataField("layout", data),
		dataset.DataField("entry", entry)])

def read_array_info(entry):
	data = try_read(files.read_array_info, "array_info", entry.array_info)
	info = recfunctions.stack_arrays([
		build_detname(data.info.det_uid, entry),
		recfunctions.drop_fields(data.info, "det_uid"),
		])
	return dataset.DataSet([
		dataset.DataField("array_info",data),
		dataset.DataField("entry", entry)])

def read_pointsrcs(entry):
	data = try_read(pointsrcs.read, "pointsrcs", entry.pointsrcs)
	return dataset.DataSet([
		dataset.DataField("pointsrcs", data),
		dataset.DataField("entry", entry)])

def read_apex(entry):
	# Get the raw weather info for the day this entry corresponds to.
	# These may be empty if the data is missing
	pwv        = try_read(files.read_apex, "pwv",  entry.pwv)
	wind_speed = try_read(files.read_apex, "wind_speed", entry.wind_speed)
	wind_dir   = try_read(files.read_apex, "wind_dir",   entry.wind_dir)
	temperature= try_read(files.read_apex, "temperature",entry.temperature)
	return dataset.DataSet([
		dataset.DataField("apex",bunch.Bunch(pwv=pwv, wind_speed=wind_speed,
		wind_dir=wind_dir, temperature=temperature))])

def read_tags(entry):
	tag_defs = try_read(files.read_tags, "tag_defs", entry.tag_defs)
	for key in tag_defs: tag_defs[key] = build_detname(tag_defs[key], entry)
	if not entry.tag:
		# If no tag was specified, we won't restrict the detectors at all,
		# so the datafield won't have a dets specification
		return dataset.DataSet([
			dataset.DataField("tag_defs", tag_defs),
			dataset.DataField("tags", [])])
	else:
		# Otherwise find the union of the tagged detectors
		tags = entry.tag.split(",")
		dets = None
		for tag in tags:
			if tag not in tag_defs:
				raise errors.DataMissing("Tag %s not defined" % (tag))
			if dets is None: dets = set(tag_defs[tag])
			else: dets &= set(tag_defs[tag])
		dets = np.array(list(dets))
		if dets.size == 0: dets = np.zeros([0],dtype="S8")
		return dataset.DataSet([
			dataset.DataField("tag_defs", tag_defs),
			dataset.DataField("tags", tags, dets=dets)]) # already expanded

def read_tod_shape(entry, moby=False):
	if moby: dets, nsamp = try_read(files.read_tod_moby, "tod_shape", entry.tod, shape_only=True)
	else:    dets, nsamp = try_read(files.read_tod,      "tod_shape", entry.tod, shape_only=True)
	return dataset.DataSet([
		dataset.DataField("tod_shape", dets=build_detname(dets, entry), samples=[0,nsamp]),
		dataset.DataField("entry", entry)])

def read_tod(entry, dets=None, moby=False, nthread=None):
	if nthread is None:
		# Too many threads is bad due to communication overhead and file system bottleneck
		nthread = min(5,int(get_dict_default(os.environ,"OMP_NUM_THREADS",5)))
	if dets is None: raw_dets = None
	else:
		# Support passing in old int det lists
		if issubclass(dets.dtype.type, np.integer): raw_dets = dets
		else: arrays, raw_dets = split_detname(dets)
	if moby: raw_dets, tod = try_read(files.read_tod_moby, "tod", entry.tod, ids=raw_dets)
	else:    raw_dets, tod = try_read(files.read_tod,      "tod", entry.tod, ids=raw_dets, nthread=nthread)
	return dataset.DataSet([
		dataset.DataField("tod", tod, dets=build_detname(raw_dets, entry), samples=[0,tod.shape[1]], det_index=0, sample_index=1, force_contiguous=True),
		dataset.DataField("entry", entry)])

readers = {
		"gain": read_gain,
		"mce_filter": read_mce_filter,
		"polangle": read_polangle,
		"tconst": read_tconst,
		"cut": read_cut,
		"point_offsets": read_point_offsets,
		"pointsrcs": read_pointsrcs,
		"array_info": read_array_info,
		"layout": read_layout,
		"beam": read_beam,
		"site": read_site,
		"noise": read_noise,
		"noise_cut": read_noise_cut,
		"spikes": read_spikes,
		"boresight": read_boresight,
		"tod_shape": read_tod_shape,
		"tod": read_tod,
		"hwp": read_hwp,
		"dark": read_dark,
		"buddies": read_buddies,
		"apex": read_apex,
		"tags": read_tags,
	}

default_fields = ["array_info","tags","beam","gain","mce_filter","polangle","tconst","cut","point_offsets","site","spikes","boresight","hwp", "pointsrcs", "buddies", "tod_shape", "tod"]
def read(entry, fields=None, exclude=None, include=None, verbose=False, dets=None):
	# Handle auto-stacking combo read transparently
	if isinstance(entry, list) or isinstance(entry, tuple):
		return read_combo(entry, fields=fields, exclude=exclude, include=include,verbose=verbose)
	# The normal case for a 1d to below
	if fields is None: fields = list(default_fields)
	if include is not None:
		for inc in include: fields.append(inc)
	if exclude is not None:
		for ex in exclude: fields.remove(ex)
	d = dataset.DataField("dummy", dets=dets)
	for field in fields:
		t1 = time.time()
		if field is "tod" and d is not None:
			d2 = readers[field](entry, dets=d.dets)
		else:
			d2 = readers[field](entry)
		if d is None: d = d2
		else: d = dataset.merge([d,d2])
		t2 = time.time()
		if verbose: print("read  %-14s in %6.3f s" % (field, t2-t1) + ("" if d.ndet is None else " %4d dets" % d.ndet))
	del d.dummy
	return d

def read_combo(entries, fields=None, exclude=None, include=None, verbose=False):
	# Read in each scan individually
	print("""FIXME: read_combo is broken. Some quantities are array-dependent but not)
detector-dependent, such as gain corrections, pointing corrections, mce gain,
mce filter parameters, point source information, beam information, etc. These
clobber each other the way things are done now. One could make it work by
going through every one of those and expanding them to be fully det dependent,
which would fix this, but this is not always straightforward, and can be a bit
wasteful, such as for the mce filter that now needs to be computed per
detector instead of being reusable. Furthermore, the detector set may also not
be available at the time of reading.

A simple way around all this is to keep everything separate until after
calibration, and only then merge things. That still leaves some properties
clobbered, but post-calibration those aren't that important any more.

A more complicated approach would be to teach DataSets about per-array quantities.
These could for example be automatically expanded to per-detector quantities as soon
as detector information becomes available, by adding them with a length-1 detector
axis. That would still require some redundant calculations but may be a good
solution.\n""")
	if len(entries) < 1: raise errors.DataMissing("Empty entry list in read_combo")
	if fields is None: fields = list(default_fields)
	if include is not None:
		for inc in include: fields.append(inc)
	if exclude is not None:
		for ex in exclude: fields.remove(ex)
	# We need array_info and boresight for combo reading
	if "array_info" not in fields: fields = ["array_info"] + fields
	if "boresight" not in fields: fields = ["boresight"] + fields
	ds = []
	for entry in entries:
		if verbose: print("reading %s" % entry.id)
		ds.append(read(entry, fields=fields, verbose=verbose))
	return merge_data(ds, verbose=verbose)

def merge_data(ds, verbose=False):
	if len(ds) == 1: return ds[0]
	# Offset samples to align them, and make detector ids unique
	#det_offs = utils.cumsum([d.array_info.ndet for d in ds])
	offs_real = measure_offsets([d.boresight[0] for d in ds])
	offs = np.round(offs_real).astype(int)
	assert np.all(np.abs(offs-offs_real) < 0.1), "Non-integer sample offset in read_combo"
	if verbose: print("offsets: " + ",".join([str(off) for off in offs]))
	if verbose: print("shifting")
	for d, off in zip(ds, offs):
		d.shift(sample_shift=off)
	# Find the common samples, as we must restrict to these before
	# we can take the union
	samples_list = np.array([d.samples for d in ds])
	samples = np.array([np.max(samples_list[:,0]),np.min(samples_list[:,1])])
	if verbose: print("restricting")
	for d in ds: d.restrict(samples=samples)
	# Ok, all datasets have the same sample range, and non-overlapping detectors.
	# Merge into a union dataset
	if verbose: print("union")
	dtot = dataset.detector_union(ds)
	# Array info cannot be automatically merged, so do it manually. We
	# assume that all have the same rectangular layout with the same number
	# of columns.
	row_offs = utils.cumsum([d.array_info.nrow for d in ds])
	infos = []
	for i, d in enumerate(ds):
		info = d.array_info.info.copy()
		#info.det_uid += det_offs[i]
		info.row     += row_offs[i]
		infos.append(info)
	info = np.rec.array(np.concatenate(infos))
	dtot.array_info.info = info
	dtot.array_info.ndet = len(info)
	dtot.array_info.nrow = np.max(info.row)+1
	# Dark detectors must also be handled manually, since they don't
	# follow the normal det slicing
	if "dark_tod"  in dtot: dtot.dark_tod  = np.concatenate([d.dark_tod for d in ds],0)
	if "dark_dets" in dtot: dtot.dark_dets = np.concatenate([d.dark_dets for i,d in enumerate(ds)],0)
	if "dark_cut"  in dtot: dtot.dark_cut = rangelist.stack_ranges([d.dark_cut for d in ds],0)
	if "tag_defs"  in dtot:
		for key in dtot.tag_defs:
			dtot.tag_defs[key] = np.concatenate([d.tag_defs[key] for i,d in enumerate(ds) if key in d.tag_defs],0)
	return dtot

def measure_offsets(times, nstep=10, dstep=1000, maxerr=0.1):
	"""Find the number of samples each timeseries in times[:,nsamp] is ahread
	of the first one."""
	times = np.array([t[0:dstep*nstep:dstep] for t in times])
	dt = np.median(times[:,1:]-times[:,:-1],1)/dstep
	# how many samples later each one starts compared to the first
	off = np.median(times-times[0],1)/dt
	return off

def require(data, fields):
	for field in fields:
		if field not in data:
			raise errors.RequireError(field)

def calibrate_boresight(data):
	"""Calibrate the boresight by converting to radians and
	interpolating across missing samples linearly. Note that
	this won't give reasonable results for gaps of length
	similar to the scan period. Also adds a srate field containing
	the sampling rate."""
	require(data, ["boresight","flags"])
	# Convert angles to radians
	if data.nsamp in [0, None]: raise errors.DataMissing("nsamp")
	if data.nsamp < 0: raise errors.DataMissing("nsamp")
	a = data.boresight[1].copy()
	bad_flag           = (data.flags!=0)*(data.flags!=0x10)
	data.boresight[1]  = robust_unwind(data.boresight[1], period=360, tol=1.0, mask=bad_flag)
	data.boresight[1:]*= np.pi/180
	#data.boresight[1:] = utils.unwind(data.boresight[1:] * np.pi/180)
	# Find unreliable regions
	bad_value  = find_boresight_jumps(data.boresight)
	bad_value |= find_elevation_outliers(data.boresight[2])
	bad = bad_flag | bad_value
	#bad += srate_mask(data.boresight[0])
	# Interpolate through bad regions. For long regions, this won't
	# work, so these should be cut.
	#  1. Raise an exception
	#  2. Construct a cut on the fly
	#  3. Handle it in the autocuts.
	# The latter is cleaner in my opinion
	cut = sampcut.from_mask(bad)
	gapfill.gapfill_linear(data.boresight, cut, inplace=True)
	srate = 1/utils.medmean(data.boresight[0,1:]-data.boresight[0,:-1])
	data += dataset.DataField("srate", srate)
	# Get the scanning speed too
	speed = calc_scan_speed(data.boresight[0], data.boresight[1])
	data += dataset.DataField("speed", speed)
	return data

def calibrate_hwp(data):
	"""Convert hwp from degrees to radians, and expand it to
	the full samples in data, since it might have dummy values if
	the hwp was not actually active for this tod."""
	require(data, ["hwp","hwp_id"])
	if data.hwp_source == "raw":
		data.hwp = data.hwp * (2*np.pi/2**16)
	else:
		data.hwp = data.hwp * utils.degree
	if data.hwp_id == "none" and data.nsamp:
		del data.hwp
		hwp = np.zeros(data.nsamp)
		data += dataset.DataField("hwp", hwp, samples=data.samples, sample_index=0)
	# Add hwp_phase, which represents the cos and sine of the hwp signal, or
	# 0 if no hwp is present
	phase = np.zeros([data.nsamp,2])
	if data.hwp_id != "none":
		phase[:,0] = np.cos(4*data.hwp)
		phase[:,1] = np.sin(4*data.hwp)
	data += dataset.DataField("hwp_phase", phase, samples=data.samples, sample_index=0)
	return data

config.default("fft_factors", "2,3,5,7,11,13", "Crop TOD lengths to the largest number with only the given list of factors. If the list includes 1, no cropping will happen.")
def crop_fftlen(data, factors=None):
	"""Slightly crop samples in order to make ffts faster. This should
	be called at a point when the length won't be futher cropped by other
	effects."""
	if data.nsamp in [0, None]: raise errors.DataMissing("nsamp")
	if data.nsamp < 0: raise errors.DataMissing("nsamp")
	factors = config.get("fft_factors", factors)
	if isinstance(factors, basestring): factors = [int(w) for w in factors.split(",")]
	ncrop = fft.fft_len(data.nsamp, factors=factors)
	data += dataset.DataField("fftlen", samples=[data.samples[0],data.samples[0]+ncrop])
	return data

def calibrate_focalplane(data):
	"""Rotate detector offsets and angles from focalplane coordinates to horizontal
	coordinates. In practice this means that we don't really support variable elevation
	scans, so this is a bit of a hack."""
	require(data, ["boresight", "point_offset"])
	el = np.mean(data.boresight[2,::100])
	if data.point_offset.size > 0:
		ocoords = coordinates.transform("bore","tele", data.point_offset.T, bore=[0,el,0,0], pol=True)
	else:
		ocoords = np.zeros([3,0])
	point_offset = ocoords[:2].T - [0,el]
	# It seems like what get get from the polarization angle file is shifted by pi/2 and
	# has the wrong sign for some reason. This could have something to do with coordinate
	# system handedness. focalplane coordinates and horizontal coordinates are left-handed,
	# while celelstial coordinates are right-handed. I flip the polarization angle rotation
	# for left-handed coordinates in coordinates.transform, and also add pi to the rotation
	# it when going between handednesses. But the inital angle is not flipped. So the
	# minus sign in front of data.polangle+pi/2 actually matches the hidden minus sign
	# ocoords[2] got inside transform due to compensating for being a left-handed coordinate
	# system. So I think this minus sign is taking us from left-handed polarization convention
	# to right-handed. This is further complicated by the distinction between the pointing
	# coordinate system and tangent plane coordinate system where the polarization is defined.
	data += dataset.DataSet([
		dataset.DataField("point_offset_raw", data.point_offset, dets=data.dets, det_index=0),
		dataset.DataField("point_offset",     point_offset,      dets=data.dets, det_index=0),
	])
	if "polangle" in data:
		polangle     = -(data.polangle + np.pi/2) + ocoords[2]
		det_comps = np.ascontiguousarray(np.array([ polangle*0+1, np.cos(+2*polangle), np.sin(2*polangle)]).T)
		data += dataset.DataSet([
			dataset.DataField("polangle_raw",     data.polangle,     dets=data.dets, det_index=0),
			dataset.DataField("polangle",         polangle,          dets=data.dets, det_index=0),
			dataset.DataField("det_comps",        det_comps,         dets=data.dets, det_index=0),
		])
	# Set up the pointing slope
	if "point_slope" in data and "site" in data:
		data.site.azslope_az0= data.point_slope[0]
		azslope_x, azslope_y = data.point_slope[1:3]
		data.site.azslope_daz, data.site.azslope_del = coordinates.bore2tele([azslope_x, azslope_y], bore=[0,el,0,0])-[0,el]
	return data

#def calibrate_point_offset(data):
#	"""Convert pointing offsets from focalplane offsets to ra,dec offsets"""
#	require(data, ["boresight", "point_offset"])
#	data.point_offset[:] = offset_to_dazel(data.point_offset, np.mean(data.boresight[1:,::100],1))
#	return data

def calibrate_buddies(data):
	"""Convert buddies to buddy_offs and buddy_comps, which describes the
	position and TQU linear combination each detector sees each buddy with.
	This assumes that boresight, polangle and point_offsets already have been
	calibrated."""
	require(data, ["buddies_raw", "boresight", "det_comps", "point_offset"])
	if data.ndet == 0: raise errors.DataMissing("ndet")
	# First build buddies from buddies_raw by adding missing detectors. We
	# do this to avoid cutting detectors that don't have any buddies, and we
	# do it here because this is where we know how many detectors there are
	if data.buddies_raw_dets is not None:
		dummy_buddy = np.zeros([0,5],float)
		buddies = np.zeros(data.ndet,object)
		for i in range(data.ndet):
			buddies[i] = dummy_buddy
		dinds, binds = utils.common_inds([data.dets, data.buddies_raw_dets])
		for dind, bind in zip(dinds, binds):
			buddies[dind] = data.buddies_raw[bind]
	else:
		# Detector-independent buddy format
		buddies = np.empty(data.ndet, object)
		for i in range(data.ndet):
			buddies[i] = data.buddies_raw[0]
	data += dataset.DataField("buddies", buddies, dets=data.dets, det_index=0)
	# Expand buddies to [nbuddy,ndet,{dx,dy,T,Q,U}]
	bfull   = expand_buddies(data.buddies, data.ndet)
	# Recover point offsets in xy plane (this would be unnecessary if
	# we handled the focalplane to horizontal conversion in the pointing matrix
	mean_bore = np.mean(data.boresight[1:,::100],1)
	raw_det_offs = dazel_to_offset(data.point_offset, mean_bore)
	# Get the buddy offsets in horizontal coordinates
	raw_buddy_offs = raw_det_offs[None] + bfull[:,:,:2]
	buddy_offs  = offset_to_dazel(raw_buddy_offs, mean_bore)
	# The buddies are modeled as only responding to T
	buddy_comps = np.zeros((len(buddy_offs),data.ndet,3))
	bfull[:,:,4] *= -1
	buddy_comps[:,:,0] = np.einsum("dc,bdc->bd", data.det_comps, bfull[:,:,2:5])
	data += dataset.DataSet([
		dataset.DataField("buddy_offs",  buddy_offs,  dets=data.dets, det_index=1),
		dataset.DataField("buddy_comps", buddy_comps, dets=data.dets, det_index=1)])
	return data

#def calibrate_polangle(data):
#	"""Rotate polarization angles to match the Healpix convention"""
#	require(data, ["polangle","boresight","point_offset"])
#	data.polangle += np.pi/2
#	# negative U component because this is the top row of a positive
#	# rotation matrix [[c,-s],[s,c]].
#	det_comps = np.ascontiguousarray(np.array([
#		data.polangle*0+1,
#		np.cos(+2*data.polangle),
#		np.sin(-2*data.polangle)]).T)
#	data += dataset.DataField("det_comps", det_comps, dets=data.dets, det_index=0)
#	return data

config.default("pad_cuts", "0:0", "Number of samples by which to widen each cut range by")
def calibrate_cut(data, n=None):
	n = [int(w) for w in config.get("pad_cuts", n).split(":")]
	for name in ["cut","cut_basic","cut_noiseest","cut_quality"]:
		if name in data:
			data[name] = data[name].widen(n)
	return data

def calibrate_beam(data):
	"""Make sure beam is equispaced. Convert radius to radians"""
	require(data, ["beam"])
	r, beam = data.beam
	assert r[0] == 0, "Beam must start from 0 radius"
	assert np.all(np.abs((r[1:]-r[:-1])/(r[1]-r[0])-1)<0.01), "Beam must be equispaced"
	data.beam = np.array([r*utils.degree, beam])
	return data

def calibrate_tod(data):
	"""Apply gain to tod and deconvolve instrument filters"""
	data = calibrate_tod_real(data)
	data = calibrate_tod_fourier(data)
	return data

config.default("simple_basic_cuts", False, "Hack: Replace basic cuts with a simple heuristic.")
def calibrate_tod_real(data, nthread=None):
	"""Apply gain to tod, fill gaps and deslope. We only gapfill
	data that's bad enough that it should be excluded when estimating
	the noise model."""
	require(data, ["tod","gain","cut_basic"])
	if data.tod.size == 0: raise errors.DataMissing("No tod samples")
	data.tod  = data.tod.astype(np.int32, copy=False)
	data.tod//= 128
	data.tod  = data.tod * (data.gain[:,None]*8)
	if config.get("simple_basic_cuts") and "cut_basic_hack" not in data:
		data += dataset.DataField("cut_basic_hack", True)
		with bench.show("simple_glitch_cut"):
			data.cut_basic = cuts.simple_glitch_cut(data.tod)
			moo1 = data.cut_basic.sum()
			moo2 = data.cut.sum()
			data.cut      *= data.cut_basic
			moo3 = data.cut.sum()
			print("simple cut cut %6.2f samples per detector, of which %6.2f were new" % (moo1/data.ndet, (moo3-moo2)/data.ndet))
	gapfill_helper(data.tod, data.cut_basic)
	utils.deslope(data.tod, w=8, inplace=True)
	return data

#config.default("fix_beam_aspect", "0:0", "Make an asymmetric beam more circular by blurring it in the scanning direction. The format is hbeam:vbeam, with each being gaussian fwhm in arcminutes. The horizontal blurring applied will bring the horizontal beam size from hbeam to vbeam. FIXME: This does not work. It appears to interact poorly with the noise model.")
def calibrate_tod_fourier(data):
	"""Deconvolve instrument filters and time constants from TOD"""
	require(data, ["tod", "tau", "srate", "speed", "mce_params"])
	if data.tod.size == 0: return data
	ft     = fft.rfft(data.tod)
	freqs  = np.linspace(0, data.srate/2, ft.shape[-1])
	# Deconvolve the butterworth filter
	butter = filters.mce_filter(freqs, data.mce_fsamp, data.mce_params)
	ft /= butter
	# And the time constants
	for di in range(len(ft)):
		ft[di] /= filters.tconst_filter(freqs, data.tau[di])
	## Optinally apply the beam aspect ratio correction
	#hbeam, vbeam = np.array(map(float,config.get("fix_beam_aspect").split(":")))*utils.arcmin*utils.fwhm
	#if vbeam != hbeam:
	#	el       = np.mean(data.boresight[2,::100])
	#	k        = 2*np.pi*freqs
	#	skyspeed = data.speed * np.cos(el)
	#	tsigma   = (vbeam**2-hbeam**2)**0.5/skyspeed
	#	ft *= np.exp(-0.5*tsigma**2*k**2)
	fft.irfft(ft, data.tod, normalize=True)
	#np.savetxt("test_enki1/tod_detau.txt", data.tod[0])
	del ft
	return data

def calibrate_dark(data):
	"""Apply gain to tod and deconvolve instrument filters"""
	data = calibrate_dark_real(data)
	data = calibrate_dark_fourier(data)
	return data

def calibrate_dark_real(data):
	"""Calibrate dark detectors. Mostly desloping."""
	#require(data, ["dark_tod","dark_cut"])
	require(data, ["dark_tod"])
	if data.dark_tod.size == 0: return data
	data.dark_tod = data.dark_tod * 1.0
	dark_cut = todops.find_spikes(data.dark_tod)
	gapfill_helper(data.dark_tod, dark_cut)
	utils.deslope(data.dark_tod, w=8, inplace=True)
	# Add the cuts to the dataset
	data += dataset.DataField("dark_cut", dark_cut, sample_index=1,
			samples=data.datafields["dark_tod"].samples)
	return data

def calibrate_dark_fourier(data):
	"""Fourier deconvolution of dark detectors. Can't do this
	completely, as we don't have time constants."""
	require(data, ["dark_tod", "srate"])
	if data.dark_tod.size == 0: return data
	ft = fft.rfft(data.dark_tod)
	freqs  = np.linspace(0, data.srate/2, ft.shape[-1])
	butter = filters.butterworth_filter(freqs)
	ft /= butter[None]
	fft.irfft(ft, data.dark_tod, normalize=True)
	return data

def calibrate_apex(data):
	"""Extract the mean of the part of the weather data relevant for
	this tod. Boresight must have been already calibrated"""
	require(data, ["apex","boresight"])
	# First replace wind direction with the wind vector. Coordinate system is
	# x-east, y-north, and indicates the direction the wind is blowing
	# *towards*, hence the minus signs.
	ispeed, idir = utils.common_inds([data.apex.wind_speed[:,0], data.apex.wind_dir[:,0]])
	wind = np.zeros([len(ispeed),3])
	wind[:,0] =  data.apex.wind_speed[ispeed,0]
	wind[:,1] = -data.apex.wind_speed[ispeed,1] * np.sin(data.apex.wind_dir[idir,1]*utils.degree)
	wind[:,2] = -data.apex.wind_speed[ispeed,1] * np.cos(data.apex.wind_dir[idir,1]*utils.degree)
	# Then extract the mean values
	period = data.boresight[0,[0,-1]]
	def extract(arr, period, mask):
		mask = mask & (arr[:,0]>=period[0])&(arr[:,0]<=period[1])
		return arr[mask,1:]
	def between(a, vmin, vmax): return (a[:,1]>=vmin)&(a[:,1]<vmax)
	data.apex.pwv  = np.mean(extract(data.apex.pwv, period, between(data.apex.pwv, 0, 50)))
	data.apex.temperature = np.mean(extract(data.apex.temperature, period, between(data.apex.temperature, -70,50)))
	wind_mask = between(data.apex.wind_speed, 0, 50)
	data.apex.wind = np.mean(extract(wind, period, wind_mask[ispeed]),0)
	data.apex.wind_speed = np.mean(extract(data.apex.wind_speed, period, wind_mask))
	# Discard wind_dir, as it does not average well. Use wind instead.
	del data.apex.wind_dir
	return data

# These just turn cuts on or off, without changing their other properties
config.default("cut_turnaround", False, "Whether to apply the turnaround cut.")
config.default("cut_ground",     False, "Whether to apply the turnaround cut.")
config.default("cut_sun",        False, "Whether to apply the sun distance cut.")
config.default("cut_moon",       False, "Whether to apply the moon distance cut.")
config.default("cut_pickup",     False, "Whether to apply the pickup cut.")
config.default("cut_obj",        "Venus,Mars,Jupiter,Saturn,Uranus,Neptune", "General list of celestial objects to cut")
config.default("cut_stationary", True,  "Whether to apply the stationary ends cut")
config.default("cut_tod_ends",   True,  "Whether to apply the tod ends cut")
config.default("cut_mostly_cut", True,  "Whether to apply the mostly cut detector cut")
# These cuts are always active, but can be effectively based on the parameter value
config.default("cut_max_frac",    0.50, "Cut whole tod if more than this fraction is autocut.")
config.default("cut_tod_mindur",  3.75, "Minimum duration of tod in minutes")
config.default("cut_tod_mindet",   100, "Minimum number of usable detectors in tod")
# These just modify the behavior of a cut. Most of these are in cuts.py
config.default("cut_sun_dist",    30.0, "Min distance to Sun in Sun cut.")
config.default("cut_moon_dist",   10.0, "Min distance to Moon in Moon cut.")
# This cut replaces the old noise whiteness cut. It probably isn't as good as that was,
# but it can be done without needing to read in the TOD, which we don't have at thi spoint
config.default("cut_tconst",    0.15, "Cut time constants longer than this number in seconds. 0 disables the cut")

config.default("cut_srcs", "map:0,nmat:0", "Point source cut. Comma-separated list of things that can be cut. Can contain map, which will produce a normal cut, or nmat which will produce a cut which is only applied during noise matrix estimation. Example. 'map:0,nmat:10000', which would not cut any sources in the map, but would ignore parts of sources brither than 10 mK when estimating the noise matrix.")

config.default("cut_extra_srcs", "", "List of [ra dec] of extra sources to cut")
config.default("cut_extra_lim",  1e-3, "Cut extra sources until they are down by this factor from beam center")

config.default("autocut",        True,  "Turn on or off all automatic cuts. Overrides their individual settings")
def autocut(d, turnaround=None, ground=None, sun=None, moon=None, max_frac=None, pickup=None):
	"""Apply automatic cuts to calibrated data."""
	if not config.get("autocut"): return d
	ndet, nsamp = d.ndet, d.nsamp
	if not ndet or not nsamp: return d
	# Insert a cut into d if necessary
	if "cut" not in d:
		d += dataset.DataField("cut", sampcut.empty(ndet,nsamp))
	# insert an autocut datafield, to keep track of how much data each
	# automatic cut cost us
	d += dataset.DataField("autocut", [])
	def addcut(label, dcut, targets="c"):
		# det ndet part here allows for broadcasting of cuts from 1-det to full-det
		dn = dcut.sum()*d.ndet/max(1,dcut.ndet) if dcut is not None else 0
		if dn == 0: d.autocut.append([label,0,0])
		else:
			n0, dn = d.cut.sum(), dcut.sum()
			dn = dn*d.cut.ndet//dcut.ndet
			if "c" in targets: d.cut *= dcut
			if "n" in targets: d.cut_noiseest *= dcut
			if "b" in targets: d.cut_basic *= dcut
			d.autocut.append([ label, dn, d.cut.sum() - n0 ]) # name, mycut, myeffect
	if config.get("cut_tconst") and "tau" in d:
		addcut("tconst", cuts.tconst_cut(nsamp, d.tau, config.get("cut_tconst")))
	if config.get("cut_stationary") and "boresight" in d:
		addcut("stationary", cuts.stationary_cut(d.boresight[1]))
	if config.get("cut_tod_ends") and "srate" in d:
		addcut("tod_ends", cuts.tod_end_cut(nsamp, d.srate))
	if config.get("cut_turnaround", turnaround) and "boresight" in d:
		addcut("turnaround",cuts.turnaround_cut(d.boresight[1]))
	if config.get("cut_ground", ground) and "boresight" in d and "point_offset" in d:
		addcut("ground", cuts.ground_cut(d.boresight, d.point_offset))
	if config.get("cut_sun", sun) and "boresight" in d and "point_offset" in d and "site" in d:
		addcut("avoidance",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Sun", config.get("cut_sun_dist")*np.pi/180))
	if config.get("cut_moon", moon) and "boresight" in d and "point_offset" in d and "site" in d:
		addcut("moon",cuts.avoidance_cut(d.boresight, d.point_offset, d.site, "Moon", config.get("cut_moon_dist")*np.pi/180))
	if config.get("cut_pickup", pickup) and "boresight" in d and "pickup_cut" in d:
		addcut("pickup",cuts.pickup_cut(d.boresight[1], d.dets, d.pickup_cut))
	if config.get("cut_mostly_cut"):
		addcut("mostly_cut", cuts.cut_mostly_cut_detectors(d.cut_quality))
	if config.get("cut_obj"):
		objs = utils.split_outside(config.get("cut_obj"),",")
		for obj in objs:
			toks = obj.split(":")
			objname = toks[0]
			if objname.startswith("["):
				objname = [float(w)*utils.degree for w in objname[1:-1].split(",")]
			# Buddies extend out to about 0.7 degrees or so in the worst case
			dist    = 0.8*utils.degree
			if len(toks) > 1: dist = float(toks[1])*utils.degree
			# Hack: only cut for noise estimation purposes if dist is negative
			targets = "cnb" if dist > 0 else "n"
			addcut(obj, cuts.avoidance_cut(d.boresight, d.point_offset, d.site, objname, dist), targets=targets)
	if config.get("cut_srcs"):
		cpar  = [tok.split(":") for tok in config.get("cut_srcs").split(",")]
		names, lims = [], []
		for par in cpar:
			if par[0] in ["map","nmat"]:
				names.append(par[0])
				lims.append(float(par[1]))
		if any(lims):
			params = pointsrcs.src2param(d.pointsrcs)
			params[:,5:7] = 1
			params[:,7]   = 0
			# Only bother with sources that are actually strong enough
			maxlim = max(lims+[0])
			params = params[params[:,2]>maxlim]
			cutlist = cuts.point_source_cut(d, params, lims)
			for name, c in zip(names, cutlist):
				if   name == "map":  addcut("point_srcs_m", c, "c")
				elif name == "nmat": addcut("point_srcs_n", c, "n")
	if config.get("cut_extra_srcs"):
		srclist = np.loadtxt(config.get("cut_extra_srcs"), usecols=(0,1), ndmin=2)
		srclim  = float(config.get("cut_extra_lim"))
		params  = np.zeros([len(srclist),8])
		params[:,:2]  = srclist[:,1::-1]*utils.degree
		params[:,2]   = 1
		params[:,5:7] = 1
		c = cuts.point_source_cut(d, params, [srclim])[0]
		addcut("point_srcs", c, "nbc")

	# What fraction is cut?
	cut_fraction = float(d.cut.sum())/d.cut.size
	# Get rid of completely cut detectors
	keep = np.where(d.cut.sum(axis=1) < nsamp)[0]
	d.restrict(d.dets[keep])
	ndet, nsamp = d.ndet, d.nsamp

	def cut_all_if(label, condition):
		if condition: dcut = sampcut.full(d.ndet, nsamp)
		else: dcut = None
		addcut(label, dcut)
	cut_all_if("max_frac",   config.get("cut_max_frac", max_frac) < cut_fraction)
	if "srate" in d:
		cut_all_if("tod_mindur", config.get("cut_tod_mindur") > nsamp/d.srate/60)
	cut_all_if("tod_mindet", config.get("cut_tod_mindet") > ndet)
	# Get rid of completely cut detectors again
	keep = np.where(d.cut.sum(axis=1) < nsamp)[0]
	d.restrict(dets=d.dets[keep])

	return d

calibrators = {
	"boresight":    calibrate_boresight,
	"gain":         calibrate_gain,
	"beam":         calibrate_beam,
	"focalplane":   calibrate_focalplane,
	"cut":          calibrate_cut,
	"autocut":      autocut,
	"fftlen":       crop_fftlen,
	"tod":          calibrate_tod,
	"tod_real":     calibrate_tod_real,
	"tod_fourier":  calibrate_tod_fourier,
	"hwp":          calibrate_hwp,
	"dark":         calibrate_dark,
	"dark_real":    calibrate_dark_real,
	"dark_foutier": calibrate_dark_fourier,
	"apex":         calibrate_apex,
	"buddies":      calibrate_buddies,
	#"polangle":     calibrate_polangle,
	#"point_offset": calibrate_point_offset,
}

default_calib = ["boresight", "gain", "focalplane", "hwp", "beam", "cut", "fftlen", "autocut", "tod_real", "tod_fourier","dark", "buddies", "apex"]
def calibrate(data, operations=None, exclude=None, strict=False, verbose=False):
	"""Calibrate the DataSet data by applying the given set of calibration
	operations to it in the given order. Data is modified inplace. If strict
	is True, then specifying a calibration operation that depends on a field
	that is not present in data raises a DataMissing exception. Otherwise,
	these are silently ignored. strict=False is useful for applying all applicable
	calibrations to a DataSet that only contains a subset of the data."""
	if operations is None: operations = list(default_calib)
	if exclude is None: exclude = []
	for ex in exclude: operations.remove(ex)
	for op in operations:
		t1 = time.time()
		status = 1
		try:
			data = calibrators[op](data)
		except errors.RequireError as e:
			if strict: raise
			status = 0
		t2 = time.time()
		if verbose: print("calib %-14s in %6.3f s" % (op, t2-t1) + (("" if data.ndet is None else " %4d dets" % data.ndet) if status else " [skipped]"))
	return data

# Helper functions

# Rz(-az)Ry(pi/2-el)Rx(y)Ry(-x): What is the meaning of this operation?
#
# Ry(-x): Rotate detector pointing from (0,0,1) to about (x,0,1)
# Rx(y):  Rotate detector pointing from (x,0,1) to about (x,y,1)
# Ry(pi/2-el): Rotate from zenith to target elevation
# Rx(-az): Rotate to target azimuth
#
# That makes sense. x and y are angular coordinates in a focalplane
# centered coordinate system, with x,y being analogous to (dec,ra).
# So if hor = [az,el] and bore=[bore_az,bore_el], then
# [y,x] = coordinates.recenter(hor, np.concatenate([bore,bore*0]))
# Yes, this works - we understand focalplane coordinates.
#
# For beam simulations I really need detector-centered coordinates.
# Approximating these as xy - xy_det should be accurate to far
# higher accuracy than our beam is known.

def offset_to_dazel(offs, azel):
	"""Convert from focalplane offsets to offsets in horizontal coordinates.
	Corresponds to the rotation Rz(-az)Ry(pi/2-el)Rx(y)Ry(-x). Taken from
	libactpol. The previous version of this was equivalent in the flat sky
	limit, but for non-tiny angles deviated by several arcseconds.
	offs should be [dx,dy] according to the data file ordering, as
	returned by the files module."""
	az, el = azel
	x, y = np.asarray(offs).T
	# Formula below based on libactpol, which uses opposite
	# ordering of y and x, so swap
	y, x = x, y
	p = [ -np.sin(x), -np.cos(x)*np.sin(y), np.cos(x)*np.cos(y) ]
	p = [ np.sin(el)*p[0]+np.cos(el)*p[2], p[1], -np.cos(el)*p[0]+np.sin(el)*p[2] ]
	dEl = np.arcsin(p[2])-el
	dAz = -np.arctan2(p[1],p[0])
	return np.array([dAz,dEl]).T

def dazel_to_offset(dazel, azel):
	"""Inverse function of offset_to_dazel"""
	az, el = azel
	dazel = np.asarray(dazel).T
	p = np.array([np.cos(-dazel[0]),np.sin(-dazel[0]),np.sin(dazel[1]+el)])
	norm = ((1-p[2]**2)/(p[0]**2+p[1]**2))**0.5
	p[:2] *= norm
	p = [np.sin(el)*p[0]-np.cos(el)*p[2],p[1], np.sin(el)*p[2]+np.cos(el)*p[0]]
	x = np.arcsin(-p[0])
	y = np.arctan2(-p[1],p[2])
	y, x = x, y
	return np.array([x,y]).T

def find_boresight_jumps(bore, width=20, tol=[1.00,0.03,0.03]):
	# median filter array to get reference behavior
	bad = np.zeros(bore.shape[-1],dtype=bool)
	width = int(width)//2*2+1
	for i, b in enumerate(bore):
		# Median filter is too slow. Let's look at blocks instead
		#fb = signal.medfilt(b, width)
		fb = utils.block_mean_filter(b, width)
		bad |= np.abs(b-fb) > tol[i]
	return bad

def find_elevation_outliers(el, tol=0.5*utils.degree):
	typ = np.median(el[::100])
	return np.abs(el-typ)>tol

config.default("gapfill", "joneig", "TOD gapfill method. Can be 'copy', 'linear' or 'cubic'")
config.default("gapfill_context", 10, "Samples of context to use for matching up edges of cuts.")
def gapfill_helper(tod, cut):
	method, context = config.get("gapfill"), config.get("gapfill_context")
	def gapfill_dummy(*args, **kwargs): pass
	gapfiller = {
			"linear":gapfill.gapfill_linear,
			"joneig":gapfill.gapfill_joneig,
			"none": gapfill_dummy,
			}[method]
	gapfiller(tod, cut, inplace=True, overlap=context)

def expand_buddies(buddies, ndet):
	"""Expand buddies to [nbuddy,ndet,{dx,dy,T,Q,U}]"""
	# Get the default buddy position for any totally missing
	# buddies. We want these to avoid placing them at [0,0], which
	# will be outside the array
	ngood, nmax, poss = 0, 0, []
	for di, buddy in enumerate(buddies):
		if len(buddy) == 0: continue
		ngood += 1
		nmax   = max(nmax,len(buddy))
		poss.append(buddy[0,:2])
	if ngood == 0: return np.zeros([0,ndet,5])
	# Initialize all buddies to [refx,refy,0,0,0]
	bfull   = np.zeros([nmax,ndet,5])
	bfull[:,:,:2] = np.mean(poss,0)
	# Then set the values for those we actually have
	for di in range(ndet):
		if len(buddies[di]) == 0: continue
		# The min and slicing here are there to accomodate the detector-independent
		# buddy format, where the array has length 1 no matter ho many dets we have.
		b = buddies[min(di,len(buddies)-1)]
		# Bfull is [nbuddy,ndet,{dx,dy,T,Q,U}]
		bfull[:len(b),di] = b
	return bfull

def robust_unwind(a, period=2*np.pi, cut=None, tol=1e-3, mask=None):
	"""Like utils.unwind, but only registers something as an angle jump if
	it is of just the right shape. If cut is specified, it should be a list
	of valid angle cut positions, which will further restrict when jumps are
	allowed. Only 1d input is supported."""
	period = float(period)
	# Find places where a jump would be acceptable. This is based on differences
	# between consecutive elements, but we ignore masked regions for this.
	# Start by constructing the masked array
	if mask is None: mask = np.full(a.shape, False, bool)
	ma     = a[~mask]
	# Then find jumps
	diffs  = (ma[1:]-ma[:-1])/period
	valid  = np.abs(np.abs(diffs)-1) < tol/period
	if cut is not None:
		near_cut = np.zeros(ma.size, bool)
		for cutval in cut:
			near_cut |= np.abs((am - cutval + period/2) % period + period/2) < tol
		valid[~near_cut] = 0
	jumps  = np.where(valid)[0]+1
	diffs  = diffs[valid]
	# Translate jumps to refer to the unmasked array
	minds  = utils.cumsum(~mask)
	jumps  = np.searchsorted(minds, jumps, side="left")
	# Then correct our values
	steps  = np.zeros(a.shape)
	steps[jumps] = np.round(diffs)
	return a - np.cumsum(steps)*period

def calc_scan_speed(t, az, step=40):
	# Quick and dirty scan speed calculation. Suffers from noise bias, but
	# should be small as long as the step isn't close to 1.
	tsub = t [::step]
	asub = az[::step]
	return utils.medmean(np.abs(asub[1:]-asub[:-1])/np.abs(tsub[1:]-tsub[:-1]))

#def build_det_group_ids(ainfo):
#	det_type = np.unique(ainfo.det_type, return_inverse=True)[1]
#	pos      = np.array([ainfo.array_x,ainfo.array_y,ainfo.nom_freq,det_type]).T
#	groups   = utils.find_equal_groups(pos, tol=1e-3)
#	group_ids = np.full([len(ainfo)],-1,int)
#	for i, g in enumerate(groups):
#		group_ids[g] = i
#	return group_ids
