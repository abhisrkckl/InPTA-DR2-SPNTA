import sys
import os
import copy
from pint.models import (
    get_model_and_toas,
    PhaseOffset,
    PLRedNoise,
    PLDMNoise,
    EcorrNoise,
)
from pint.models.parameter import maskParameter
from pint.fitter import Fitter
from astropy import units as u
import numpy as np

psrname = sys.argv[1]
datadir = f"InPTA.DR2/{psrname}"
parfile_in = f"InPTA.DR2/{psrname}/{psrname}.DMX.par"
timfile = f"InPTA.DR2/{psrname}/{psrname}_all.tim"

outdir = f"analysis/{psrname}"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

model_in, toas = get_model_and_toas(parfile_in, timfile, allow_tcb=True)
parfile_out = f"{outdir}/{psrname}_spnta.par"
timfile_out = f"{outdir}/{psrname}_all.tim"
toas.write_TOA_file(timfile_out)  # To merge all tim files into one.

model_out = copy.deepcopy(model_in)

Tspan = toas.get_Tspan()
Tcad_red = 0.25 * u.year

# Add phase offset
model_out.add_component(PhaseOffset())
model_out["PHOFF"].frozen = False

# Add red noise
model_out.add_component(PLRedNoise())
model_out["TNREDAMP"].value = -14
model_out["TNREDGAM"].value = 3.5
model_out["TNREDC"].value = int(
    np.ceil((Tspan / Tcad_red / 2).to_value(u.dimensionless_unscaled))
)
model_out["TNREDFLOG"].value = 4
model_out["TNREDFLOG_FACTOR"].value = 2

# Replace DMX by DMGP
N_DMX = len(model_out.components["DispersionDMX"].get_indices())
N_DM = 2
model_out.remove_component("DispersionDMX")
model_out.add_component(PLDMNoise())
model_out["DM"].frozen = False
model_out["DM1"].value = 0
model_out["DM1"].frozen = False
model_out["TNDMAMP"].value = -14
model_out["TNDMGAM"].value = 3.5
model_out["TNDMC"].value = int(np.ceil((N_DMX - N_DM) / 2))
model_out["TNDMFLOG"].value = 4
model_out["TNDMFLOG_FACTOR"].value = 2

# Add EQUADs and ECORRs
model_out.add_component(EcorrNoise())
for efacname in model_out.components["ScaleToaError"].EFACs:
    efac: maskParameter = model_out[efacname]
    idx = efac.index

    equadname = f"EQUAD{idx}"
    if idx == 1:
        equad = model_out[equadname]
        equad.value = 1e-3
    else:
        equad = maskParameter(
            name=equadname,
            index=idx,
            key=efac.key,
            key_value=efac.key_value,
            value=1e-3,
            units=model_out["EQUAD1"].units,
            description=model_out["EQUAD1"].description,
            tcb2tdb_scale_factor=1,
        )
        model_out.components["ScaleToaError"].add_param(equad)

    ecorrname = f"ECORR{idx}"
    if idx == 1:
        ecorr = model_out[ecorrname]
        ecorr.value = 1e-13
    else:
        ecorr = maskParameter(
            name=ecorrname,
            index=idx,
            key=efac.key,
            key_value=efac.key_value,
            value=1e-3,
            units=model_out["ECORR1"].units,
            description=model_out["ECORR1"].description,
            tcb2tdb_scale_factor=1,
        )
        model_out.components["EcorrNoise"].add_param(ecorr)

# Do a preliminary fit for "cheat" priors. This need not be accurate.
ftr = Fitter.auto(toas, model_out)
try:
    ftr.fit_toas(maxiter=20)
except:
    # The downhill fitter sometimes throws spurious errors.
    pass
model_out = ftr.model

# Unfreeze noise parameters
model_out["TNREDAMP"].frozen = False
model_out["TNREDGAM"].frozen = False
model_out["TNDMAMP"].frozen = False
model_out["TNDMGAM"].frozen = False
for efacname in model_out.components["ScaleToaError"].EFACs:
    model_out[efacname].frozen = False
for equadname in model_out.components["ScaleToaError"].EQUADs:
    model_out[equadname].frozen = False
for ecorrname in model_out.components["EcorrNoise"].ECORRs:
    model_out[ecorrname].frozen = False

model_out.write_parfile(parfile_out)
