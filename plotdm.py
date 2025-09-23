#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform
from pint import DMconst, dmu
from pint import logging
from pyvela import SPNTA
import astropy.units as u

logging.setup(level="ERROR")

psrname = sys.argv[1]
dmdatadir = f"InPTA.DR2/DM-timeseries/"
dmfile3 = f"{dmdatadir}/{psrname}_DM_timeseries.B3.txt"
dmdata3 = np.genfromtxt(dmfile3)
dmfile35 = f"{dmdatadir}/{psrname}_DM_timeseries.B35.txt"
dmdata35 = np.genfromtxt(dmfile35) if os.path.isfile(dmfile35) else None

resultdir = f"analysis/{psrname}/results/"
parfile = f"{resultdir}/{psrname}.median.par"
timfile = f"{resultdir}/{psrname}_all.tim"
m = get_model(parfile)
m.remove_component("ScaleToaError")
m.remove_component("EcorrNoise")
# m.remove_param("PLREDFREQ")
# m.remove_param("PLDMFREQ")
tsim = make_fake_toas_uniform(
    startMJD=np.min(dmdata3[:,0]),
    endMJD=np.max(dmdata3[:,0]),
    ntoas=1000,
    model=m,
    add_noise=False,
    add_correlated_noise=False,
)
freqs_sim = m.barycentric_radio_freq(tsim)
mjds = tsim.get_mjds()

spnta = SPNTA.load_jlso(f"analysis/{psrname}/results/_{psrname}.jlso", parfile, timfile)
spnta_sim = SPNTA.from_pint(m, tsim)

plt.errorbar(dmdata3[:,0], dmdata3[:,1], dmdata3[:,2], ls="", marker="+", color="blue", label="DMcalc (B3)")
if dmdata35 is not None:
    plt.errorbar(dmdata35[:,0], dmdata35[:,1], dmdata35[:,2], ls="", marker="+", color="red", label="DMcalc (B3+B5)")
ylim = plt.ylim()

samples_raw = np.load(f"analysis/{psrname}/results/samples_raw.npy")
samples = np.load(f"analysis/{psrname}/results/samples.npy")
pnames = np.genfromtxt(f"analysis/{psrname}/results/param_names.txt", dtype=str)
pnames_marg = np.genfromtxt(f"analysis/{psrname}/results/marginalized_param_names.txt", dtype=str)
dmgp_idxs = np.array([idx for idx, pname in enumerate(pnames_marg) if pname.startswith("PLDMSIN_" or pname.startswith("PLDMCOS_"))])
M_dmgp = np.array(spnta_sim.model.kernel.noise_basis)[:, dmgp_idxs]

d_D_d_DM = m.d_dm_d_param(tsim, "DM")
d_D_d_DM1 = m.d_dm_d_param(tsim, "DM1")
if "NE_SW" in m.free_params:
    d_D_d_NESW = m.d_dm_d_param(tsim, "NE_SW")

dm_idx = list(pnames).index("DM")
dm1_idx = list(pnames).index("DM1")
if "NE_SW" in m.free_params:
    nesw_idx = list(pnames).index("NE_SW")

selection = np.random.choice(np.arange(samples_raw.shape[0]), 100, replace=False)
for idx in selection:
    pvals_raw = samples_raw[idx, :]
    pvals_marg = spnta.get_marginalized_param_sample(pvals_raw)
    pvals = samples[idx, :]
    a_dmgp = pvals_marg[dmgp_idxs]

    dm = (
        d_D_d_DM * pvals[dm_idx] * m["DM"].units 
        + d_D_d_DM1 * pvals[dm1_idx] * m["DM1"].units 
        + (M_dmgp @ a_dmgp) * u.s * freqs_sim**2 / DMconst
    ).to_value(dmu)

    if "NE_SW" in m.free_params:
        dm += (d_D_d_NESW * pvals[nesw_idx] * m["NE_SW"].units).to_value(dmu)

    plt.plot(tsim.get_mjds().value, dm, color="k", alpha=0.05)

plt.ylim(ylim)
plt.legend()
plt.xlabel("MJD")
plt.ylabel("DM (pc/cm^3)")
plt.title(m["PSR"].value)
plt.tight_layout()
plt.show()




