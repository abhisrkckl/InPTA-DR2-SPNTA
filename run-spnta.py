#!/usr/bin/env python

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
from pint.logging import setup as setup_log
from pint.fitter import Fitter
from astropy import units as u
import numpy as np
import json
import utils
import subprocess

setup_log(level="WARNING")

psrname = sys.argv[1]
datadir = f"InPTA.DR2/{psrname}"
parfile_in = f"InPTA.DR2/{psrname}/{psrname}.DMX.par"
timfile = f"InPTA.DR2/{psrname}/{psrname}_all.tim"

outdir = f"analysis/{psrname}"
if not os.path.isdir(outdir):
    os.mkdir(outdir)

model_in, toas = get_model_and_toas(parfile_in, timfile, allow_tcb=True)
model_in = model_in.as_ICRS()
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
    equad1 = model_out["EQUAD1"]
    if idx == 1:
        equad = model_out[equadname]
        equad.from_parfile_line(f"EQUAD {efac.key} {efac.key_value[0]} 1e-3 0")
    else:
        equad = maskParameter(
            name=equadname,
            index=idx,
            tcb2tdb_scale_factor=1
        )
        equad.from_parfile_line(f"EQUAD {efac.key} {efac.key_value[0]} 1e-3 0")
        model_out.components["ScaleToaError"].add_param(equad)

    ecorrname = f"ECORR{idx}"
    if idx == 1:
        ecorr = model_out[ecorrname]
        ecorr.from_parfile_line(f"ECORR {efac.key} {efac.key_value[0]} 1e-3 0")
    else:
        ecorr = maskParameter(
            name=ecorrname,
            index=idx,
            tcb2tdb_scale_factor=1
        )
        ecorr.from_parfile_line(f"ECORR {efac.key} {efac.key_value[0]} 1e-3 0")
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

# Unfreeze astrometric parameters.
# We have already converted to ICRS coordinates if necessary.
model_out["PX"].frozen = False
model_out["PMRA"].frozen = False
model_out["PMDEC"].frozen = False

# Write modified par file
model_out.write_parfile(parfile_out)

# Prepare the prior JSON file.
px_dm = utils.get_px_from_dm(model_out)
prior_dict = {
    "PMRA": {
        "distribution": "PGeneralizedGaussian",
        "args": [0.3608272992359529, -1.9900000477866313, 0.3313620365404956],
        "source": "psrcat"
    },
    "PMDEC": {
        "distribution": "PGeneralizedGaussian",
        "args": [-1.9900000477866313, 0.3313620365404956, 0.3608272992359529],
        "source": "psrcat"
    },
    "PX": {
        "distribution": "Normal",
        "args": [px_dm, px_dm/2],
        "lower": 0.0,
        "source": "pygedm[ymw16]"
    }
}

astrometry_data = utils.get_astrometry_data(psrname)
if astrometry_data is not None:
    raj, σraj, decj, σdecj, pmra, σpmra, pmdec, σpmdec, px, σpx, src = astrometry_data
    prior_dict.update(
        {
            "RAJ": {
                "distribution": "Normal",
                "args": [raj.value, 100*σraj.value],
                "lower": 0.0,
                "upper": 24.0,
                "source": src,
            },
            "DECJ": {
                "distribution": "Normal",
                "args": [decj.value, 100*σdecj.value],
                "lower": -90.0,
                "upper": 90.0,
                "source": src,
            },
            "PMRA": {
                "distribution": "Normal",
                "args": [pmra.value, σpmra.value],
                "source": src,
            },
            "PMDEC": {
                "distribution": "Normal",
                "args": [pmdec.value, σpmdec.value],
                "source": src,
            },
            "PX": {
                "distribution": "Normal",
                "args": [px.value, σpx.value],
                "lower": 0.0,
                "source": src,
            },
        }
    )

prior_file = f"{outdir}/{psrname}_priors.json"
with open(prior_file, "w") as f:
    json.dump(prior_dict, f, indent=4)


# Run the analysis
result_dir = f"{outdir}/results/"
pyvela_cmd = f"pyvela {parfile_out} {timfile_out} -P {prior_file} -o {result_dir} -A all -C 100 -f"
print(f"Running command :: {pyvela_cmd}")
subprocess.run(pyvela_cmd.split())