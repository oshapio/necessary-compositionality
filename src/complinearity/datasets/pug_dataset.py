import os

import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image


class PUGDataset(data.Dataset):
    """PUG Animal dataset loader."""

    CONCEPT_COLUMNS = (
        "world-name",
        "character-name",
        "character-scale",
        "character-texture",
        "camera-yaw",
    )
    MAX_VALUES_PER_CONCEPT = {
        "world-name": 64,
        "character-name": 69,
        "character-scale": 3,
        "character-texture": 4,
        "camera-yaw": 4,
    }

    def __init__(
        self,
        csv_path,
        images_folder,
        transform=None,
    ):
        super().__init__()
        df = pd.read_csv(csv_path)
        if "character_name" in df.columns:
            df = df[df["character_name"] != "Goldfish"]
        df.columns = [col.replace("_", "-") for col in df.columns]
        self.df = df.reset_index(drop=True)

        self.images_folder = images_folder
        self.transform = transform
        self.cols_to_consider = list(self.CONCEPT_COLUMNS)
        self.rebuild_maps_and_ids()

    def rebuild_maps_and_ids(self):
        self.col_name_val_to_idx = {}
        self.col_idx_to_name_val = {}
        for col in self.df.columns:
            uniques = list(pd.unique(self.df[col]))
            self.col_name_val_to_idx[col] = {val: idx for idx, val in enumerate(uniques)}
            self.col_idx_to_name_val[col] = {idx: val for idx, val in enumerate(uniques)}

        for col in self.cols_to_consider:
            if col in self.df.columns:
                self.df[f"{col}-id"] = self.df[col].map(self.col_name_val_to_idx[col])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.at[index, "filename"]
        character_name = self.df.at[index, "character-name"]
        image_path = os.path.join(self.images_folder, character_name, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        attrs_tensor = torch.tensor(
            [self.df.at[index, f"{col}-id"] for col in self.CONCEPT_COLUMNS],
            dtype=torch.long,
        )

        return {
            "image": image,
            "index": index,
            "image_path": image_path,
            "filename": filename,
            "attrs": attrs_tensor,
        }

    def dictify_attrs_ids(self, names=None):
        """Create nested dict keyed by attribute ids; leaves hold sample indices."""
        attr_order = list(names) if names is not None else list(self.CONCEPT_COLUMNS)
        attr_values = self.df[attr_order].to_numpy(copy=False)

        dictified_attrs = {}
        for i, row in enumerate(attr_values):
            current_dict = dictified_attrs
            for j, attr_name in enumerate(attr_order):
                attr_val = row[j]
                attr_id = self.col_name_val_to_idx[attr_name][attr_val]
                is_leaf = j == len(attr_order) - 1
                if is_leaf:
                    current_dict.setdefault(attr_id, []).append(i)
                else:
                    current_dict = current_dict.setdefault(attr_id, {})

        def get_num_samples(dct):
            if isinstance(dct, dict):
                return sum(get_num_samples(v) for v in dct.values())
            if isinstance(dct, list):
                return len(dct)
            raise RuntimeError("Unexpected type in dictify_attrs_ids")

        assert len(self.df) == get_num_samples(dictified_attrs), "Mismatch between df length and dictified samples"
        return dictified_attrs

    @staticmethod
    def concept_names_and_ids_in_order():
        return {
            "name_to_pos": {name: i for i, name in enumerate(PUGDataset.CONCEPT_COLUMNS)},
            "max_vals_per_concept": dict(PUGDataset.MAX_VALUES_PER_CONCEPT),
        }


class RestrictedPUGDataset(PUGDataset):
    """PUGDataset with simple include/exclude filters on columns."""

    def __init__(
        self,
        csv_path,
        images_folder,
        filter_conditions=None,
        exclude_conditions=None,
        transform=None,
    ):
        super().__init__(csv_path, images_folder, transform)

        if filter_conditions:
            for column, value in filter_conditions.items():
                self.df = self.df[self.df[column] == value]
        if exclude_conditions:
            for column, value in exclude_conditions.items():
                self.df = self.df[self.df[column] != value]
        self.df = self.df.reset_index(drop=True)
        self.rebuild_maps_and_ids()
