import os

import numpy as np
import torch
from torch.utils.data import Dataset


HEAD_TO_IDX = {
    "color": 0,
    "shape": 1,
    "size": 2,
    "orientation": 3,
    "posx": 4,
    "posy": 5,
}


class CleanDSpritesDataset(Dataset):
    """Minimal loader for clean dSprites `.npz` data."""

    def __init__(self, npz_path, transform=None, load_as_float=False, **_ignored):
        if not os.path.exists(npz_path):
            raise FileNotFoundError(npz_path)
        data = np.load(npz_path, allow_pickle=True)

        self.images = data["imgs"]
        self.latents_classes = data["latents_classes"]
        self.latents_values = data["latents_values"]

        meta = data["metadata"].item()
        self.latents_sizes = np.asarray(meta["latents_sizes"], dtype=int)

        self.transform = transform
        self.load_as_float = bool(load_as_float)
        self.base_len = int(self.images.shape[0])

    def __len__(self):
        return self.base_len

    def __getitem__(self, index):
        img = self.images[index]
        if not self.load_as_float:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
        image_out = self.transform(img) if self.transform is not None else img
        attrs = torch.tensor(self.latents_classes[index], dtype=torch.long)
        return {
            "image": image_out,
            "index": int(index),
            "attrs": attrs,
        }

    def labels_y_for_heads(self, heads=None, expand_colors=True):
        if heads is None:
            heads = list(HEAD_TO_IDX.keys())
        y = {}
        for h in heads:
            idx = HEAD_TO_IDX[h]
            y[h] = self.latents_classes[:, idx]
        return y

    def captions_bank_and_values(self, heads=None):
        if heads is None:
            heads = list(HEAD_TO_IDX.keys())
        values_bank = {}
        captions_bank = {}
        for head in heads:
            idx = HEAD_TO_IDX[head]
            num_values = int(self.latents_sizes[idx])
            values = list(range(num_values))
            values_bank[head] = values
        return {"values": values_bank, "captions": captions_bank}
