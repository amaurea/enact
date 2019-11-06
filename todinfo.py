from __future__ import division, print_function
import numpy as np
from enlib import coordinates, utils, errors, bunch, tagdb, ephemeris
from enact import actdata, files
day_range = [11,23]
jon_ref   = 1378840304

tsys = "hor"

# Implement our todinfo database using a Tagdb
class Todinfo(tagdb.Tagdb):
	def __init__(self, data, sort="id"):
		tagdb.Tagdb.__init__(self, data, sort=sort, default_fields=["sel",("t",np.NaN)], default_query="sel,isfinite(t)")
		# Define wrapper here for the default argument
		self.add_functor("hits",   hits_fun)
		self.add_functor("dist",   dist_fun)
		self.add_functor("grow",   grow_fun)
		self.add_functor("esplit", esplit_fun)
		self.add_functor("planet", planet_fun)
		self.add_functor("elements", elements_fun)
		self.add_functor("hor",    hor_fun)
	# Print the most useful fields + the true tags for each tod
	def __repr__(self, nmax=None):
		lines = []
		n = len(self)
		n1, n2 = (n, 0) if not nmax or n <= nmax else (nmax//4, nmax//4)
		finfo = [
				("id",  "%25s", "%25s"),
				("sel", "%3d",   "%3s"),
				("hour","%5.2f", "%5s"),
				("az",  "%7.2f", "%7s"),
				("el",  "%6.2f", "%6s"),
				("ra",  "%6.2f", "%6s"),
				("dec", "%6.2f", "%6s"),
				("pwv", "%5.2f", "%5s"),
				("wx",  "%6.2f", "%6s"),
				("wy",  "%6.2f", "%6s"),
		]
		# Prune so we can still print something if the normal fields are missing
		finfo = [fi for fi in finfo if fi[0] in self.data]
		fdata = [self.data[fi[0]] for fi in finfo]
		lfmt  = " " + " ".join([fi[1] for fi in finfo])
		hfmt  = "#" + " ".join([fi[2] for fi in finfo])
		hnames= tuple([fi[0] for fi in finfo])
		# Ok, generate the output lines
		header= hfmt % hnames + " tags"
		lines = [header]
		def pline(i):
			line = lfmt % tuple([fd[i] for fd in fdata])
			line += " " + " ".join(sorted([key for key,val in self.data.iteritems() if key != "id" and val.dtype == bool and val.ndim == 1 and val[i]]))
			return line
		for i in range(0,n1):
			lines.append(pline(i))
		if n2 > 0:
			lines.append("       ...       ")
			for i in range(n-n2, n):
				lines.append(pline(i))
		return "\n".join(lines)
	def __str__(self): return self.__repr__(100)
	@classmethod
	def read_txt(cls, fname, vars={}):
		"""Read a Tagdb from text files, supporting Loic's selelected tod format"""
		datas = []
		for subfile, tags in tagdb.parse_tagfile_top(fname, vars=vars):
			data = parse_tagfile_loic(subfile)
			# tags with format :tag result in tag being appended to the
			# id:subtag,subtag,.... Then tag is added to the normal list of tags
			subid  = ",".join([tag[1:] for tag in tags if tag.startswith(":")])
			tags   = [tag.lstrip(":") for tag in tags]
			for tag in tags:
				data[tag] = np.full(len(data["id"]), True, dtype=bool)
			data["id"] = tagdb.append_subs(data["id"], subid)
			datas.append(data)
		res = cls(tagdb.merge(datas))
		return res

def parse_tagfile_loic(fname):
	ids = []
	sel = []
	with open(fname,"r") as f:
		for line in f:
			line = line.rstrip()
			if len(line) < 1 or line[0] == "#": continue
			toks = line.split()
			# There are two formats
			if len(toks) >= 6:
				# 1. [id] [hour] [alt] [az] [pwv] [cut status] [[tag]]
				# Here a tod is only sel if the status is 2
				# Hack: work around corrupt lines
				if len(toks[0].split(".")) != 3: continue
				ids.append(toks[0])
				sel.append(int(toks[5]) == 2)
			elif len(toks) == 1:
				# 2. path to file. Here we will extract the id from
				# the path, and mark everything as *not* selectd.
				# So /all will be needed to access these.
				id = toks[0].split("/")[-1]
				if id.endswith(".zip"): id = id[:-4]
				# Hack: work around corrupt lines
				if len(id.split(".")) != 3: continue
				ids.append(id)
				sel.append(False)
	ids = np.asarray(ids + ["foo"])[:-1]
	sel = np.asarray(sel)
	return {"id":ids, "sel":sel}

def read(fname, type=None, vars={}):
	return Todinfo.read(fname, type, vars=vars)

# Functions that can be used in todinfo queries
# Coordinate order is ra,dec
def point_in_polygon_safe(points, polygons):
	points   = np.asarray(points)
	polygons = np.array(polygons)
	# Put the polygon on the same side of the sky as the points
	polygons[0] = utils.rewind(polygons[0], points[0], 360)
	# But don't allow sky wraps inside polygons
	polygons[0] = utils.rewind(polygons[0], polygons[0,0], 360)
	return utils.point_in_polygon(points.T, polygons.T)
def grow_polygon(polys, dist):
	print("FIXME: grow_polygon is wrong")
	polys = np.array(polys)
	dist  = np.zeros(2) + dist
	# Compensate for curvature
	dist_eff = polys.copy()
	dist_eff[0] = dist[0] / np.cos(polys[1])
	dist_eff[1] = dist[1]
	# Expand away from center independently in each dimension
	mid = np.mean(polys,1)
	for i in range(2):
		polys[i,polys[i]<mid[i]] -= dist[i]
		polys[i,polys[i]>mid[i]] += dist[i]
	return polys
def poly_dist(points, polygons):
	points   = np.asarray(points)*utils.degree
	polygons = np.array(polygons)*utils.degree
	# Put the polygon on the same side of the sky as the points
	polygons[0] = utils.rewind(polygons[0], points[0])
	# But don't allow sky wraps inside polygons
	polygons[0] = utils.rewind(polygons[0], polygons[0,0])
	inside = utils.point_in_polygon(points.T, polygons.T)
	dists  = utils.poly_edge_dist(points.T, polygons.T)
	dists  = np.where(inside, 0, dists)
	return dists

class hits_fun:
	def __init__(self, data): self.data = data
	def __call__(self, point, polys=None, tol=0.5):
		if polys is None: polys = self.data["bounds"]
		if tol == 0: return point_in_polygon_safe(point, polys)
		else:        return poly_dist(point, polys) < tol*utils.degree
class dist_fun:
	def __init__(self, data): self.data = data
	def __call__(self, point, ref=None):
		if ref is None: ref=[self.data["ra"],self.data["dec"]]
		return utils.angdist(np.array(point)*utils.degree,np.array(ref)*utils.degree, zenith=False)/utils.degree
class grow_fun:
	def __init__(self, data): self.data = data
	def __call__(self, polys, dist):
		return grow_polygon(polys*utils.degree, dist*utils.degree)/utils.degree
class esplit_fun:
	"""Select the ind'th split out of nsplit total splits of
	of the data using a day-wise greedy split that tries to
	get as equal amount of data as possible in each split."""
	def __init__(self, data): self.data = data
	def __call__(self, nsplit, ind, nday=4):
		# First find the number of tods in each day group
		dayind = (np.asarray(self.data["jon"])/nday).astype(int)
		uind, inv, counts = np.unique(dayind, return_inverse=True, return_counts=True)
		groups = utils.greedy_split_simple(counts, nsplit)
		group_sel = np.zeros(len(counts), np.bool)
		group_sel[groups[ind]] = True
		tod_sel = group_sel[inv]
		return tod_sel
class planet_fun:
	def __init__(self, data): self.data = data
	def __call__(self, name):
		mjd = utils.ctime2mjd(self.data["t"])
		mjd[~np.isfinite(mjd)] = 0
		pos = coordinates.ephem_pos(name, mjd)
		pos /= utils.degree
		return pos
class elements_fun:
	def __init__(self, data): self.data = data
	def __call__(self, fname):
		mjd = utils.ctime2mjd(self.data["t"])
		mjd[~np.isfinite(mjd)] = 0
		obj = ephemeris.read_object(fname)
		pos = ephemeris.ephem_pos(obj, mjd)[:2]
		pos /= utils.degree
		return pos
class hor_fun:
	def __init__(self, data): self.data = data
	def __call__(self, pos):
		mjd = utils.ctime2mjd(self.data["t"])
		hor = coordinates.transform("cel","hor", pos*utils.degree, mjd)/utils.degree
		return hor

# Functions for extracting tod stats from tod files. Useful for building
# up Todinfos.
def build_tod_stats(entry, Naz=8, Nt=2):
	"""Collect summary information for the tod in the given entry, returning
	it as a bunch. If some information can't be found, then those fields will
	be set to a placeholder value (usually NaN), but the fields will still all
	be present."""
	# At the very least we need the pointing, so no try catch around this
	d = actdata.read(entry, ["boresight","site"])
	d += actdata.read_point_offsets(entry, no_correction=True)
	d = actdata.calibrate(d, exclude=["autocut"])

	# Get the array center and radius
	acenter = np.mean(d.point_offset,0) 
	arad    = np.mean((d.point_offset-acenter)**2,0)**0.5

	t, baz, bel = 0.5*(np.min(d.boresight,1)+np.max(d.boresight,1))
	#t, baz, bel = np.mean(d.boresight,1)
	az  = baz + acenter[0]
	el  = bel + acenter[1]
	dur, waz, wel = np.max(d.boresight,1)-np.min(d.boresight,1)
	mjd  = utils.ctime2mjd(t)
	hour = t/3600.%24
	day   = hour >= day_range[0] and hour < day_range[1]
	night = not day
	jon   = (t - jon_ref)/(3600*24)

	ra, dec = coordinates.transform(tsys,"cel",[az,el],mjd, site=d.site)
	# Get the array center bounds on the sky, assuming constant elevation
	ts  = utils.ctime2mjd(t+dur/2*np.linspace(-1,1,Nt))
	azs = az + waz/2*np.linspace(-1,1,Naz)
	E1 = coordinates.transform(tsys,"cel",[azs,         [el]*Naz],time=[ts[0]]*Naz, site=d.site)[:,1:]
	E2 = coordinates.transform(tsys,"cel",[[azs[-1]]*Nt,[el]*Nt], time=ts,          site=d.site)[:,1:]
	E3 = coordinates.transform(tsys,"cel",[azs[::-1],   [el]*Naz],time=[ts[-1]]*Naz,site=d.site)[:,1:]
	E4 = coordinates.transform(tsys,"cel",[[azs[0]]*Nt, [el]*Nt], time=ts[::-1],    site=d.site)[:,1:]
	bounds = np.concatenate([E1,E2,E3,E4],1)
	bounds[0] = utils.rewind(bounds[0])
	## Grow bounds by array radius
	#bmid = np.mean(bounds,1)
	#for i in range(2):
	#	bounds[i,bounds[i]<bmid[i]] -= arad[i]
	#	bounds[i,bounds[i]>bmid[i]] += arad[i]
	tot_id = entry.id + (":" + entry.tag if entry.tag else "")
	res = bunch.Bunch(id=tot_id, nsamp=d.nsamp, t=t, mjd=mjd, jon=jon,
			hour=hour, day=day, night=night, dur=dur,
			az =az /utils.degree,  el =el/utils.degree,
			baz=baz/utils.degree,  bel=bel/utils.degree,
			waz=waz/utils.degree,  wel=wel/utils.degree,
			ra =ra /utils.degree,  dec=dec/utils.degree,
			bounds = bounds/utils.degree)

	# Planets
	for obj in ["Sun","Moon","Mercury","Venus","Mars","Jupiter","Saturn","Uranus","Neptune"]:
		res[obj] = coordinates.ephem_pos(obj, utils.ctime2mjd(t))/utils.degree

	# Get our weather information, if available
	try:
		d += actdata.read(entry, ["apex"])
		d  = actdata.calibrate_apex(d)
		res["pwv"] = d.apex.pwv
		res["wx"] = d.apex.wind[0]
		res["wy"] = d.apex.wind[1]
		res["wind_speed"] = d.apex.wind_speed
		res["T"] = d.apex.temperature
	except errors.DataMissing:
		res["pwv"] = np.NaN
		res["wx"] = np.NaN
		res["wy"] = np.NaN
		res["wind_speed"] = np.NaN
		res["T"] = np.NaN
	
	# Try to get our cut info, so that we can select on
	# number of detectors and cut fraction
	try:
		npre = d.nsamp*d.ndet
		d += actdata.read(entry, ["cut"])
		res["ndet"] = d.ndet
		res["cut"] = 1-d.nsamp*d.ndet/float(npre)
	except errors.DataMissing:
		res["ndet"] = 0
		res["cut"] = 1.0

	# Try to get hwp info
	res["hwp"] = False
	res["hwp_name"] = "none"
	try:
		epochs = actdata.try_read(files.read_hwp_epochs, "hwp_epochs", entry.hwp_epochs)
		t, _, ar = entry.id.split(".")
		t = float(t)
		if ar in epochs:
			for epoch in epochs[ar]:
				if t >= epoch[0] and t < epoch[1]:
					res["hwp"] = True
					res["hwp_name"] = epoch[2]
	except errors.DataMissing:
		pass

	return res

def merge_tod_stats(statlist):
	return bunch.Bunch(**{key: np.array([stat[key] for stat in statlist]) for key in statlist[0]})

def get_tods(selector, db):
	try:
		return np.array(utils.read_lines(selector))
	except IOError:
		return db[selector]
