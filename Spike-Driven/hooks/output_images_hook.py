# hooks/output_images_hook.py
# 2025-05-16  wyjung: supports custom color-map + input capture + safer RGB grids

import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib import cm

__all__ = ["OutputImageSaver"]


class OutputImageSaver:
    """
    Collects layer outputs via the `hook_dict` you already build during forward
    and saves (1) raw .pt tensors, (2) color heat-maps, (3) grid images.

    Example
    -------
    saver = OutputImageSaver(
        base_dir       = "…/hook_images",
        sample_batches = [0, 100, 156],      # which batches to dump
        cmap           = "viridis",
    )

    …
    # inside training / validation loop
    saver.dump(batch_idx, hook_dict, raw_input=batch_input)
    """

    def __init__(
        self,
        base_dir: str,
        sample_batches=(0,),
        cmap: str = "viridis",
    ):
        self.base_dir = base_dir
        self.sample_batches = set(sample_batches)
        self.cmap = cm.get_cmap(cmap)
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def dump(self, batch_idx: int, hook_dict: dict, raw_input=None):
        if batch_idx not in self.sample_batches:
            return

        if raw_input is not None:
            self._save_input(batch_idx, raw_input)

        for name, tensor in hook_dict.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            self._save_tensor(batch_idx, name, tensor.cpu())

    # ------------------------------------------------------------------ #
    def _save_input(self, idx: int, img: torch.Tensor):
        img_dir = os.path.join(self.base_dir, "input_images")
        os.makedirs(img_dir, exist_ok=True)
        # assume NCHW float in [0,1] or normalised – denorm for viewing if desired
        vutils.save_image(img[0].float().cpu(), os.path.join(img_dir, f"b{idx}_input.png"))

    def _save_tensor(self, idx: int, name: str, tensor: torch.Tensor):
        lay_dir = os.path.join(self.base_dir, name.replace("/", "_"))
        os.makedirs(lay_dir, exist_ok=True)

        # --- 1 · raw tensor ---------------------------------------------------
        torch.save(tensor, os.path.join(lay_dir, f"b{idx}.pt"))

        # --- 2 · colour heat-map ---------------------------------------------
        arr = tensor.detach().float().numpy()

        # collapse to ≤ 2-D               (mean over T, C, N if present)
        while arr.ndim > 2:
            arr = arr.mean(axis=0)

        if arr.ndim == 1:                      # (L,) → (1,L) bar
            arr = arr[None, :]

        fig = plt.figure(figsize=(4, 4))
        plt.imshow(arr, cmap=self.cmap, aspect="auto")
        plt.title(name)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(os.path.join(lay_dir, f"b{idx}_heatmap.png"))
        plt.close(fig)

        # --- 3 · grid image (1st sample, up to 3 channels) --------------------
        if tensor.ndim == 4:                   # (N,C,H,W)
            vis = tensor[0].clone()
            if vis.size(0) == 1:
                vis = vis.repeat(3, 1, 1)
            elif vis.size(0) > 3:
                vis = vis[:3]
            vutils.save_image(
                vis,
                os.path.join(lay_dir, f"b{idx}_grid.png"),
                normalize=True,
                scale_each=True,
            )
