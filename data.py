"""This module provides higher-level access to actpol data input, operating on
fildb entries instead of filenames, and reading all the information for you."""
from enact import files, errors
from enlib import zgetdata, utils
from bunch import Bunch # use a simple bunch for now

def read(entry):
	"""Given a filedb entry, reads all the data associated with it.
	Only detectors for which all the information is present will be
	returned, and missing files will raise a DataMissing error."""
	for key in ["gain","gain_correction","polangle","cut","tconst","point_template","point_offsets","site","tod"]:
		if entry[key] is None:
			raise errors.DataMissing("Missing %s in entry for %s" % (key,entry.id))
	try:
		# Perform all the scary read operations
		id_gain, gain = files.read_gain(entry.gain)
		gain *= files.read_gain_correction(entry.gain_correction)[entry.id]
		id_pol,  pol  = files.read_polangle(entry.polangle)
		id_tau,  tau  = files.read_tconst(entry.tconst)
		id_cut,  cut  = files.read_cut(entry.cut)
		id_off,  off  = files.read_point_template(entry.point_template)
		off  += files.read_point_offsets(entry.point_offsets)[entry.id][:,None]
		site          = files.read_site(entry.site)
	except IOError  as e: raise DataMissing("%s [%s]" % (e.message, entry.id))
	except KeyError as e: raise DataMissing("Gain correction or pointing offset [%s]" % entry.id)
	# Restrict to common set of ids
	ind_gain, ind_pol, ind_tau, ind_cut, ind_off = utils.common_inds([id_gain,id_pol,id_tau,id_cut,id_off])
	gain, pol, tau, cut, off = gain[ind_gain], pol[ind_pol], tau[ind_tau], cut[ind_cut], off[:,ind_off]
	id = id_gain[ind_gain]
	# Then get the boresight and time-ordered data
	try:
		with zgetdata.dirfile(entry.tod) as dfile:
			bore, flags = files.read_boresight(dfile)
			id,   tod   = files.read_tod(dfile, id)
	except zgetdata.OpenError as e:
		raise errors.DataMissing(e.message + "[%s]" % entry.id)
	return Bunch(gain=gain, polangle=pol, tconst=tau, cut=cut, point_offsets=off, tod=tod, boresight=bore, enc_flags=flags, dets=id, entry=entry, site=site)
