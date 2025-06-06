import csv, os, math
from pathlib import Path
from spikingjelly.activation_based import functional
from syops import get_model_complexity_info


class MetricsLogger:
    """
    Long-format CSV logger: epoch, metric, tag, value
    """

    def __init__(self, csv_path, input_size, ac_pj=0.9, mac_pj=4.6):
        self.csv_path   = csv_path
        self.input_size = input_size          # (C, H, W)
        self.ac_pj, self.mac_pj = ac_pj, mac_pj

        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "metric", "tag", "value"])

    # ───────────────────────────────── helpers
    def _write(self, epoch, metric, tag, value):
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, metric, tag, value])

    # ───────────────────────────────── public API
    def log_top1(self, epoch, top1):
        self._write(epoch, "top1", "overall", f"{top1:.4f}")
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([])

    def log_syops(self, epoch, model, dataloader=None):
        functional.reset_net(model)
        syops, params = get_model_complexity_info(
            model, self.input_size, dataloader=dataloader,
            as_strings=False, print_per_layer_stat=False, verbose=False
        )
        total, acs, macs = syops
        energy = (acs * self.ac_pj + macs * self.mac_pj) * 1e-9   # pJ → mJ
        self._write(epoch, "syops",  "total", total)
        self._write(epoch, "syops",  "ACs",   acs)
        self._write(epoch, "syops",  "MACs",  macs)
        self._write(epoch, "energy", "mJ",    f"{energy:.3f}")
        self._write(epoch, "params", "count", params)

    def log_firing(self, epoch, hook_dict):
        for name, spikes in hook_dict.items():
            # 4 significant figures
            val = spikes.mean().item()
            self._write(epoch, "firing", name, f"{val:.4g}")

    def log_gradnorm(self, epoch, module_list):
        for i, block in enumerate(module_list):
            norm_sq = sum((p.grad.data.norm(2).item() ** 2)
                          for p in block.parameters() if p.grad is not None)
            self._write(epoch, "grad", f"encoder_{i}", math.sqrt(norm_sq))
