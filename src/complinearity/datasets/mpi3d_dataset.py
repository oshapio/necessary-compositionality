import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MPI3DDataset(Dataset):
    """MPI3D loader with a small dSprites-like API."""

    FACTORS_IN_ORDER = [
        "object-color",
        "object-shape",
        "object-size",
        "camera-height",
        "background-color",
        "horizontal-axis",
        "vertical-axis",
    ]
    FACTOR_TO_INDEX = {name: idx for idx, name in enumerate(FACTORS_IN_ORDER)}

    NPZ_IMAGE_KEYS = ("images", "imgs", "X", "data")
    NUM_VALUES_PER_FACTOR = [6, 6, 2, 3, 3, 40, 40]

    def __init__(self, data_path, transform=None, pos_keep_bins=10, load_as_float=False):
        self.data_path = data_path
        self.load_as_float = bool(load_as_float)
        self.transform = transform

        self.images, labels = self.load_images_and_labels(data_path, load_as_float=self.load_as_float)

        self.factor_sizes = list(self.NUM_VALUES_PER_FACTOR)
        self._validate_image_count()
        self.mapper = self._build_mapper(labels)

        self.base_len = int(self.images.shape[0])
        self._df = None

        if isinstance(pos_keep_bins, int) and pos_keep_bins > 0:
            self.downsample_axes(pos_keep_bins)

    def _validate_image_count(self):
        expected = int(np.prod(self.factor_sizes))
        count = int(self.images.shape[0])
        if count != expected:
            raise ValueError(
                f"Image count {count} does not match expected grid size {expected} "
                f"for factor sizes {self.factor_sizes}. Update NUM_VALUES_PER_FACTOR if needed."
            )

    def _build_mapper(self, labels):
        if labels is not None:
            mapper = np.asarray(labels)
            if mapper.ndim != 2 or mapper.shape[1] != len(self.FACTORS_IN_ORDER):
                raise ValueError(f"labels must be (N, {len(self.FACTORS_IN_ORDER)}), got {mapper.shape}")
            return mapper

        grid = tuple(self.factor_sizes)
        mapper = np.indices(grid).reshape(len(grid), -1).T
        if mapper.shape[0] != self.images.shape[0]:
            raise ValueError("images count does not match product of fixed factor sizes")
        return mapper

    @classmethod
    def _extract_image_array_from_npz(cls, data):
        for key in cls.NPZ_IMAGE_KEYS:
            if key in data:
                return np.asarray(data[key])
        keys = list(data.keys())
        if len(keys) == 1:
            return np.asarray(data[keys[0]])

    @staticmethod
    def _to_float01(arr):
        arr = arr.astype(np.float32, copy=False)
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr

    def load_images_and_labels(self, path, load_as_float=False):
        """Load images (and labels if present) from directory or npz file."""
        labels = None
        if os.path.isfile(path):
            if path.endswith(".npz"):
                data = np.load(path, allow_pickle=True)
                arr = self._extract_image_array_from_npz(data)
                if "labels" in data:
                    labels = data["labels"]
            elif path.endswith(".npy"):
                arr = np.load(path, allow_pickle=True)
        else:
            images_npy = os.path.join(path, "images.npy")
            if not os.path.exists(images_npy):
                raise FileNotFoundError(images_npy)
            arr = np.load(images_npy, allow_pickle=True)

        arr = np.asarray(arr)
        if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (0, 2, 3, 1))
        if load_as_float:
            arr = self._to_float01(arr)
        if labels is not None:
            labels = np.asarray(labels)
        return arr, labels

    @staticmethod
    def _spaced_bins(n_bins, keep_n):
        keep_n = int(min(max(1, keep_n), n_bins))
        if keep_n >= n_bins:
            return np.arange(n_bins, dtype=int)
        if keep_n <= 1:
            return np.array([0], dtype=int)
        step = (n_bins - 1) / float(keep_n - 1)
        return np.round(np.arange(keep_n, dtype=float) * step).astype(int)

    def downsample_axes(self, keep_n):
        """Keep rows whose horizontal/vertical ids are in equally spaced bins."""
        idx_h = self.FACTOR_TO_INDEX["horizontal-axis"]
        idx_v = self.FACTOR_TO_INDEX["vertical-axis"]
        bins_h = int(self.factor_sizes[idx_h])
        bins_v = int(self.factor_sizes[idx_v])

        allowed_h = self._spaced_bins(bins_h, keep_n)
        allowed_v = self._spaced_bins(bins_v, keep_n)

        keep_mask = np.isin(self.mapper[:, idx_h], allowed_h) & np.isin(self.mapper[:, idx_v], allowed_v)
        if np.all(keep_mask):
            return

        self.images = self.images[keep_mask]
        self.mapper = self.mapper[keep_mask]
        self.base_len = int(keep_mask.sum())
        self._df = None

    def __len__(self):
        return self.base_len

    def __getitem__(self, index):
        if index < 0 or index >= self.base_len:
            raise IndexError
        img = self.images[index]
        if not self.load_as_float:
            img = self._to_float01(img)
        image_out = self.transform(img) if self.transform is not None else img
        attrs = torch.as_tensor(self.mapper[index], dtype=torch.long)
        return {
            "image": image_out,
            "index": int(index),
            "attrs": attrs,
        }

    def dataframe(self):
        import pandas as pd

        if self._df is not None:
            return self._df.copy()
        cols = list(self.FACTORS_IN_ORDER)
        df = pd.DataFrame(self.mapper, columns=cols)
        for c in cols:
            df[c + "-id"] = df[c]
        df["index"] = np.arange(self.base_len)
        self._df = df
        return df.copy()

    def head_to_col(self):
        return {h: h for h in self.FACTORS_IN_ORDER}

    def labels_y_for_heads(self, heads=None, expand_colors=False):
        df = self.dataframe()
        if heads is None:
            heads = list(self.FACTORS_IN_ORDER)
        return {
            h: df[f"{h}-id"].to_numpy(dtype=np.int32)
            for h in heads
            if f"{h}-id" in df.columns
        }

    @staticmethod
    def _caption_for(head, value):
        if head in ("horizontal-axis", "vertical-axis"):
            return f"A photo with {head.replace('-', ' ')} bin {value}"
        if head == "object-size":
            label = {0: "small", 1: "large"}.get(value, "object")
            return f"A photo of a {label} object"
        if head == "camera-height":
            label = {0: "top", 1: "center", 2: "bottom"}.get(value, "unknown")
            return f"A photo from {label} camera height"
        return f"A photo with {head.replace('-', ' ')} {value}"

    def captions_bank_and_values(self, heads=None):
        if heads is None:
            heads = list(self.FACTORS_IN_ORDER)

        values = {}
        captions = {}
        for h in heads:
            idx = self.FACTOR_TO_INDEX.get(h)
            if idx is None:
                continue
            uniq = np.unique(self.mapper[:, idx]).astype(int).tolist()
            values[h] = uniq
            captions[h] = [self._caption_for(h, v) for v in uniq]
        return {"values": values, "captions": captions}

