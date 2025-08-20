import pygedm
from pint.models import TimingModel
from astropy.coordinates import Angle
from astropy import units as u
from astropy.table import Table
import numpy as np

def get_px_from_dm(model: TimingModel):
    coords = model.coords_as_GAL()
    dm = model["DM"].value
    D: u.Quantity = pygedm.dm_to_dist(coords.l, coords.b, dm, method="ymw16")[0]
    return (u.AU / D).to_value("mas", equivalencies=u.dimensionless_angles())





def get_astrometry_data(psrname):
    data_colnames = ["PSR",  "RAJ", "σRAJ", "DECJ", "σDECJ", "PMRA", "σPMRA", "PMDEC", "σPMDEC", "PX", "σPX",  "Source"]
    data_dtypes = ["U15", "U20", float, "U20", float, float, float, float, float, float, float, "U15"]
    astrometry_data = Table(
        np.genfromtxt("astrometry.txt", comments="#", dtype=data_dtypes),
        names=data_colnames,
    )

    if psrname not in astrometry_data["PSR"]:
        return None

    idx = list(astrometry_data["PSR"]).index(psrname)
    raj, σraj, decj, σdecj, pmra, σpmra, pmdec, σpmdec, px, σpx, src = astrometry_data[idx][data_colnames[1:]]
    return (
        Angle(f"{raj} hours"),
        u.Quantity(σraj, u.mas).to(u.hourangle),
        Angle(f"{decj} degrees"),
        u.Quantity(σdecj, u.mas).to(u.degree),
        u.Quantity(pmra, u.mas/u.yr),
        u.Quantity(σpmra, u.mas/u.yr),
        u.Quantity(pmdec, u.mas/u.yr),
        u.Quantity(σpmdec, u.mas/u.yr),
        u.Quantity(px, u.mas),
        u.Quantity(σpx, u.mas),
        src,
    )