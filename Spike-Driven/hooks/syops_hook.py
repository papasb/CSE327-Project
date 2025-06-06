# hooks/syops_hook.py
import torch
import numpy as np          
from syops import get_model_complexity_info

def profile_model(model, input_size, dataloader=None,
                  ac_pj=0.9, mac_pj=4.6, as_strings=False):
    """
    Returns a dict with total/AC/MAC SyOps, params, and simple energy estimate.
    """
    syops, params = get_model_complexity_info(
        model, input_size,
        dataloader=dataloader,
        as_strings=as_strings,
        print_per_layer_stat=False,
        verbose=False,
    )

    if as_strings:                       # already human-readable
        total, acs, macs = syops
        energy_mJ = "n/a"
    else:                                # numerical → compute energy
        total, acs, macs = syops         # float values
        energy_mJ = (acs * ac_pj + macs * mac_pj) * 1e-9  # pJ → mJ

    return {
        "TotalSyOps": total,
        "ACs":        acs,
        "MACs":       macs,
        "Params":     params,
        "Energy_mJ":  energy_mJ,
    }