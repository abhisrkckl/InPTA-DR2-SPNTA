import pygedm
from pint.models import TimingModel
from astropy.coordinates import SkyCoord
from astropy import units as u

def get_px_from_dm(model: TimingModel):
    coords = model.coords_as_GAL()
    dm = model["DM"].value
    D: u.Quantity = pygedm.dm_to_dist(coords.l, coords.b, dm, method="ymw16")[0]
    return (u.AU / D).to_value("mas", equivalencies=u.dimensionless_angles())
    