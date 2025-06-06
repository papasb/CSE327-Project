from collections import defaultdict
from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode

def attach_spike_tap(model, collector):
    """
    Register a forward-hook on every LIF / PLIF in model.
    Each hook writes its spike tensor into collector with a unique key.
    """
    
    name_cnt = defaultdict(int)
    
    def _make_hook(base):
        def _hook(_, __, out):
            key = f"{base}_{name_cnt[base]}"
            collector[key] = out.detach()
        return _hook
    
    for name, module in model.named_modules():
        if isinstance(module, (LIFNode, ParametricLIFNode)):
            module.register_forward_hook(_make_hook(name))
            name_cnt[name] += 1
        