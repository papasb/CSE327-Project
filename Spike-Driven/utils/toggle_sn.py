# utils/toggle_sn.py

# 05/03 (wyjung): Ablation Study #1: Disable SN (LIF) in different stages of the network and compare metrics.

import torch
import logging
_logger = logging.getLogger("train")

def disable_lif(module: torch.nn.Module):
    import types
    
    # replace forward with identity
    module.forward = types.MethodType(lambda self, x: x, module)
    
def apply_toggle(model, cfg):
    """
    Method to disable SN (LIF) in different stages of the network according to three switches S0-S2.
    - S0: All LIFs in MS_SPS disabled
        - Measures how much temporal sparsity the visual frontend contributes.
    - S1: Only LIF in RPE disabled
        - Is the extra spiking non-linearity after the residual helpful?
    - S2: Keep SPS & RPE LIF, remove LIF in encoder (SDSA & MLP) blocks
        - Does most performance come from convolutional spikes, or do we also need spiking token mixing?
    """
    if not cfg.S0:      # leave SPS as-is
        pass
    else:               # disable all LIFs in MS_SPS
        _logger.info('S0: Disable all LIFs in MS_SPS.')
        for m in model.modules():
            if m.__class__.__name__.startswith('MS_SPS'):
                for attr in ['proj_lif', 'proj_lif1', 'proj_lif2', 'proj_lif3']:
                    disable_lif(getattr(m, attr))
                    
    if cfg.S1:
        _logger.info('S1: Disable LIF in RPE.')
        for m in model.modules():
            if hasattr(m, 'rpe_lif'):
                disable_lif(m.rpe_lif)
                
    if cfg.S2:
        _logger.info('S2: Disable LIF in SDSA & MLP blocks.')
        for m in model.modules():
            if m.__class__.__name__.endswith('Block'):
                disable_lif(m.lif_q)
                disable_lif(m.lif_k)
                disable_lif(m.lif_v)
                disable_lif(m.mlp_lif)
                