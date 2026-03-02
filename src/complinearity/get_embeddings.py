import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data

from complinearity import _config
from complinearity.embedding_utils import (
    build_output_path,
    load_backend,
    model_signature,
    normalize_pretrained_input,
    sibling_with_ext,
)

from tqdm.auto import tqdm

from complinearity.datasets.mpi3d_dataset import MPI3DDataset
from complinearity.datasets.pug_dataset import RestrictedPUGDataset
from complinearity.datasets.clean_dsprites_dataset import CleanDSpritesDataset
from complinearity.datasets.dsprites_dataset import DSpritesDataset

DSPRITES_NPZ_PATH = _config.npz_path_dsprites
DSPRITES_CLEAN_NPZ_PATH = _config.npz_path_dsprites_clean
MPI3D_NPZ_PATH = _config.npz_path_mpi3d
PUG_CSV_PATH = _config.pug_csv_path
PUG_IMAGES_FOLDER = _config.pug_images_folder

DSPRITES_NUM_COLORS = 10
DSPRITES_ORIENTATION_KEEP_DEG = (0.0, 90.0)
DEFAULT_POS_KEEP_BINS = 10

PUG_FILTER_CONDITIONS = {"camera-yaw": 0}
PUG_EXCLUDE_CONDITIONS = {"character-texture": "Default"}

def run_embeddings(
    dataset,
    output_path,
    backend="none",
    model_name="ViT-B/32",
    pretrained=None,
    device_str="cuda",
    batch_size=64,
    num_workers=8,
    caption_types=None,
    save_pkl=True,
    pkl_output_path=None,
    resume=True,
    random_init=False,
    random_seed=None,
    random_std=0.02,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    backend = str(backend).strip().lower()
    is_dsprites = DSpritesDataset is not None and isinstance(dataset, DSpritesDataset)
    is_mpi3d = isinstance(dataset, MPI3DDataset)
    is_clean_dsprites = isinstance(dataset, CleanDSpritesDataset)

    model, preprocess, used_pretrained_tag = load_backend(
        backend=backend,
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        random_init=random_init,
        random_seed=random_seed,
        random_std=random_std,
    )

    # Ensure preprocess handles numpy HWC (dSprites) by wrapping via PIL
    def wrap_preprocess_for_numpy(preprocess_fn):
        def apply_preprocess(x):
            if isinstance(x, np.ndarray):
                x8 = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
                return preprocess_fn(Image.fromarray(x8))
            return preprocess_fn(x)
        return apply_preprocess

    if preprocess is None:
        dataset.transform = None
    elif is_dsprites or is_mpi3d or is_clean_dsprites:
        dataset.transform = wrap_preprocess_for_numpy(preprocess)
    else:
        dataset.transform = preprocess

    # Build output paths early and support resume
    if is_dsprites:
        dataset_label = "dsprites"
    elif is_clean_dsprites:
        dataset_label = "clean_dsprites"
    elif is_mpi3d:
        dataset_label = "mpi3d"
    else:
        dataset_label = "pug"
    use_model = model is not None
    use_cat_pixels = (backend == "cat")
    if use_cat_pixels and not (is_dsprites or is_clean_dsprites):
        raise ValueError("backend='cat' is currently supported only for dsprites and clean_dsprites")
    if use_model:
        model.eval()
    model_sig = (
        model_signature(backend, model_name, used_pretrained_tag)
        if (use_model or use_cat_pixels)
        else "none"
    )
    path_pretrained_tag = used_pretrained_tag
    output_file = build_output_path(output_path, backend, model_name, path_pretrained_tag, dataset_label)
    default_pkl_file = sibling_with_ext(output_file, ".pkl")
    pkl_path = pkl_output_path or default_pkl_file
    # if resume and (os.path.exists(pkl_path) or os.path.exists(output_file)):
    #     print(f"[resume] Found existing outputs. Skipping compute. npz={output_file} pkl={pkl_path if os.path.exists(pkl_path) else 'missing'}")
    #     return

    loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if num_workers > 0 else False,
    )

    all_img_embeds = []
    total_params = None
    if use_model:
        total_params = int(sum(p.numel() for p in model.parameters()))
    printed_model_info = False
    if caption_types is None:
        if is_dsprites:
            caption_types = ["shape", "size", "orientation", "posx", "posy", "color"]
        elif is_clean_dsprites:
            caption_types = ["color", "shape", "size", "orientation", "posx", "posy"]
        elif is_mpi3d:
            caption_types = [
                "object-color",
                "object-shape",
                "object-size",
                "camera-height",
                "background-color",
                "horizontal-axis",
                "vertical-axis",
            ]
        else:
            caption_types = ["world", "character", "size", "texture"]

    filenames = []
    values_bank = {}
    captions_bank = {}
    y_indices = {}

    if is_dsprites or is_mpi3d or is_clean_dsprites:
        banks = dataset.captions_bank_and_values(heads=caption_types)
        values_bank = banks["values"]
        captions_bank = banks["captions"]
        y_indices = dataset.labels_y_for_heads(heads=caption_types, expand_colors=True)
    else:
        df = dataset.df
        head_to_col = {
            "world": "world-name",
            "character": "character-name",
            "size": "character-scale",
            "texture": "character-texture",
        }
        for head, col in head_to_col.items():
            if head in caption_types and (col + "-id") in df.columns:
                y_indices[head] = df[col + "-id"].to_numpy(dtype=np.int32)
        for head, col in head_to_col.items():
            if head not in caption_types or col not in df.columns:
                continue
            mapping = getattr(dataset, "col_idx_to_name_val", {}).get(col)
            if isinstance(mapping, dict) and len(mapping) > 0:
                max_idx = max(mapping.keys())
                values = [mapping[i] for i in range(max_idx + 1)]
            else:
                values = list(pd.unique(df[col]))
            values_bank[head] = values
            caps = []
            if head == "world":
                for v in values:
                    caps.append(f"A photo in a {str(v).lower()} environment")
            elif head == "character":
                for v in values:
                    caps.append(f"A photo of a {str(v).lower()}")
            elif head == "size":
                for v in values:
                    caps.append(f"A photo of an object at scale {v}")
            elif head == "texture":
                for v in values:
                    if str(v) == "Default":
                        caps.append("A photo of an object")
                    else:
                        caps.append(f"A photo of an object textured with {str(v).lower()}")
            captions_bank[head] = caps

    inference_ctx = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_ctx():
        for batch in tqdm(loader, total=len(loader), desc="Embedding", dynamic_ncols=True):
            # Images
            if use_model:
                images = batch["image"].to(device, non_blocking=True)
                img_feats = model.encode_image(images)
                img_feats = img_feats.float()
                if not printed_model_info:
                    emb_dim = int(img_feats.shape[-1]) if img_feats.ndim >= 2 else int(img_feats.numel())
                    msg = f"[model] backend={backend} name={model_name} emb_dim={emb_dim}"
                    if total_params is not None:
                        msg += f" params={total_params}"
                    print(msg)
                    printed_model_info = True
                # Save RAW (non-normalized) image embeddings for probing
                img_feats_np = img_feats.detach().cpu().numpy()
                all_img_embeds.append(img_feats_np)
            elif use_cat_pixels:
                images = batch["image"]
                if isinstance(images, np.ndarray):
                    images_t = torch.from_numpy(images)
                else:
                    images_t = images
                if not isinstance(images_t, torch.Tensor):
                    raise RuntimeError(f"Unexpected image batch type for cat backend: {type(images)}")
                # Expected shape from datasets without preprocess: (B, H, W, C).
                # If CHW sneaks in, flatten still remains valid.
                images_t = images_t.float().max(dim=3, keepdim=True).values
                flat = images_t.reshape(images_t.shape[0], -1)
                if not printed_model_info:
                    emb_dim = int(flat.shape[-1])
                    print(f"[model] backend={backend} name={model_name} emb_dim={emb_dim}")
                    printed_model_info = True
                all_img_embeds.append(flat.cpu().numpy())
            # else:
            #     # If no model, infer batch size from filenames or indices
            #     if "filename" in batch:
            #         batch_size_curr = len(batch["filename"])
            #     elif "index" in batch:
            #         indices_val = batch["index"]
            #         batch_size_curr = len(indices_val if isinstance(indices_val, list) else indices_val)
            #     else:
            #         batch_size_curr = 0

            if "filename" in batch:
                filenames.extend(list(batch["filename"]))
            elif "index" in batch:
                idxs = batch["index"].cpu().numpy().tolist() if hasattr(batch["index"], "cpu") else list(batch["index"])
                filenames.extend([f"idx_{int(i)}" for i in idxs])

    # Stack and save
    npz_kwargs = {"filenames": np.array(filenames, dtype=object)}
    if all_img_embeds:
        npz_kwargs["image_embeds"] = np.concatenate(all_img_embeds, axis=0)
    # Save labels/values banks and indices
    for head in caption_types:
        # # Always save values and y if available
        if head in values_bank:
            npz_kwargs[f"values_{head}"] = np.array(values_bank.get(head, []), dtype=object)
        if head in y_indices:
            npz_kwargs[f"y_{head}"] = y_indices[head]

    # Save minimal metadata for reproducibility
    meta = {
        "backend": backend,
        "model_name": model_name,
        "pretrained": pretrained,
        "random_init": bool(random_init),
        "random_seed": random_seed,
        "random_std": float(random_std),
        "caption_types": caption_types,
        "num_samples": len(dataset),
        "model_signature": model_sig,
        "image_embeds_normalized": False,
    }
    meta["dataset"] = "dsprites" if is_dsprites else ("mpi3d" if is_mpi3d else ("clean_dsprites" if is_clean_dsprites else "pug"))
    meta["path_pretrained_tag"] = path_pretrained_tag
    npz_kwargs["meta_json"] = np.array([json.dumps(meta)], dtype=object)

    np.savez_compressed(output_file, **npz_kwargs)

    # Optional pickle with richer content (banks + indices)
    if save_pkl:
        pkl_path = pkl_output_path or sibling_with_ext(output_file, ".pkl")
        pkl_data = {
            "filenames": filenames,
            "meta": meta,
            "values": values_bank,
            "captions_bank": captions_bank,
            "y": y_indices,
        }
        if all_img_embeds:
            pkl_data["image_embeds"] = np.concatenate(all_img_embeds, axis=0)
        os.makedirs(os.path.dirname(pkl_path) or ".", exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[saved] npz={output_file}")
    if save_pkl:
        print(f"[saved] pkl={pkl_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pug", "dsprites", "mpi3d", "clean_dsprites"],
        default="clean_dsprites",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/clip_models_laion",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="cat",
        choices=["none", "clip", "openclip", "dino", "timm", "cat"],
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="cat",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion400m_e32",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument(
        "--random-init",
        action="store_true",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--random-std",
        type=float,
        default=0.02,
    )

    args = parser.parse_args()
    if str(args.backend).strip().lower() == "cat":
        # Keep default runs readable in filenames/W&B config without needing extra flags.
        if not args.model_name or str(args.model_name).strip() in {"ViT-B-32", "ViT-B/32"}:
            args.model_name = "cat"

    filter_conditions = dict(PUG_FILTER_CONDITIONS)
    exclude_conditions = dict(PUG_EXCLUDE_CONDITIONS)

    # Instantiate selected dataset; transforms are injected later
    if args.dataset == "dsprites":
        if DSpritesDataset is None:
            raise ValueError("Dataset 'dsprites' is unavailable because complinearity.datasets.dsprites_dataset is missing.")
        dataset = DSpritesDataset(
            npz_path=DSPRITES_NPZ_PATH,
            use_colors=True,
            num_colors=DSPRITES_NUM_COLORS,
            orientation_keep_deg=DSPRITES_ORIENTATION_KEEP_DEG,
            pos_keep_bins=DEFAULT_POS_KEEP_BINS,
            transform=None,
        )
    elif args.dataset == "clean_dsprites":
        dataset = CleanDSpritesDataset(
            npz_path=DSPRITES_CLEAN_NPZ_PATH,
            transform=None,
        )
    elif args.dataset == "mpi3d":
        dataset = MPI3DDataset(
            data_path=MPI3D_NPZ_PATH,
            transform=None,
            pos_keep_bins=DEFAULT_POS_KEEP_BINS,
        )
    else:
        dataset = RestrictedPUGDataset(
            csv_path=PUG_CSV_PATH,
            images_folder=PUG_IMAGES_FOLDER,
            filter_conditions=filter_conditions if filter_conditions else None,
            exclude_conditions=exclude_conditions if exclude_conditions else None,
            transform=None,
        )

    run_embeddings(
        dataset=dataset,
        output_path=args.output_path,
        backend=args.backend,
        model_name=args.model_name,
        pretrained=normalize_pretrained_input(args.pretrained, args.backend),
        device_str=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_init=bool(args.random_init),
        random_seed=args.random_seed,
        random_std=float(args.random_std),
    )


if __name__ == "__main__":
    main()
