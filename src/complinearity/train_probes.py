import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from tqdm.auto import tqdm


def login_wandb():
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key


login_wandb()
def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pkl":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Pickle payload must be a dict")
        return payload

    if ext == ".npz":
        data = np.load(path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    raise ValueError("Unsupported format; use .pkl or .npz")

def get_output_paths(
    emb_path,
    out_dir,
    lr,
    random_init=False,
    random_seed=None,
):
    """Return `(probes_json_path, probes_prefix_path)` for an embeddings file."""
    emb_stem = os.path.splitext(os.path.basename(emb_path))[0]
    if random_init and "_rand" not in emb_stem:
        seed_suffix = f"_s{random_seed}" if random_seed is not None else ""
        emb_stem = f"{emb_stem}_rand{seed_suffix}"

    probes_dir = out_dir or os.path.join(os.path.dirname(emb_path), "_probes")
    os.makedirs(probes_dir, exist_ok=True)

    lr_tag = f"lr{str(lr).replace('.', 'p')}"
    probes_name = f"_probes_{emb_stem}_{lr_tag}"
    probes_prefix_path = os.path.join(probes_dir, probes_name)
    probes_json_path = f"{probes_prefix_path}.json"
    return probes_json_path, probes_prefix_path


@torch.no_grad()
def accuracy(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())


class MultiHeadProbes(nn.Module):
    def __init__(self, in_dim, head_to_num_classes):
        super().__init__()
        self.heads = nn.ModuleDict({h: nn.Linear(in_dim, c, bias=True) for h, c in head_to_num_classes.items()})

    def forward(self, x):
        return {h: layer(x) for h, layer in self.heads.items()}


def train_all_heads_joint(
    X,
    head_to_y,
    head_to_values,
    lr,
    weight_decay,
    epochs,
    batch_size,
    seed,
    val_split=0.05,
    log_every=10,
    head_name_prefix="",
    cosine_decay=False,
    eta_min=0.0,
):
    """
    Train all concept heads in a single loop with one optimizer.
    Uses full CE per head and sums losses; shared feature matrix X.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    in_dim = int(X.shape[1])
    num_total = int(X.shape[0])

    # Build tensors on device
    X_t = torch.from_numpy(X).to(device, non_blocking=True)

    heads = list(head_to_y.keys())
    head_to_num_classes = {h: int(np.max(y) + 1) for h, y in head_to_y.items()}
    model = MultiHeadProbes(in_dim, head_to_num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=eta_min) if cosine_decay else None

    # Split indices once for all heads (fixed seed for determinism)
    split_gen = torch.Generator()
    split_gen.manual_seed(seed)
    idx = torch.randperm(num_total, generator=split_gen)
    n_val = int(val_split * num_total)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    head_y_train = {h: torch.from_numpy(y[train_idx.cpu().numpy()].astype(np.int64)).to(device, non_blocking=True) for h, y in head_to_y.items()}
    head_y_val = {h: torch.from_numpy(y[val_idx.cpu().numpy()].astype(np.int64)).to(device, non_blocking=True) for h, y in head_to_y.items()} if n_val > 0 else {}

    best_state = {h: None for h in heads}
    best_val = {h: float('-inf') for h in heads}

    X_train = X_t[train_idx]
    X_val = X_t[val_idx] if n_val > 0 else None

    pbar = tqdm(range(1, epochs + 1), desc="Train (joint)", dynamic_ncols=True)
    for ep in pbar:
        model.train()
        total_seen = 0
        total_loss = 0.0
        epoch_gen = torch.Generator(device=device)
        epoch_gen.manual_seed(seed + ep)
        perm = torch.randperm(X_train.shape[0], generator=epoch_gen, device=device)
        # Iterate batches on train split
        for start in range(0, X_train.shape[0], batch_size):
            end = min(start + batch_size, X_train.shape[0])
            batch_idx = perm[start:end]
            xb = X_train[batch_idx]
            batch_targets = {h: head_y_train[h][batch_idx] for h in heads}

            optimizer.zero_grad()
            logits_dict = model(xb)
            loss_val = torch.zeros((), device=xb.device)
            for h in heads:
                loss_val = loss_val + criterion(logits_dict[h], batch_targets[h])
            loss_val.backward()
            optimizer.step()
            loss_scalar = float(loss_val.detach().item())

            bs = int(batch_idx.numel())
            total_seen += bs
            total_loss += loss_scalar * bs

        # Validation every log_every epochs
        if n_val > 0 and (ep % log_every == 0 or ep == epochs):
            model.eval()
            with torch.no_grad():
                logits_val = model(X_val)
                for h in heads:
                    acc = accuracy(logits_val[h], head_y_val[h])
                    if acc > best_val[h]:
                        best_val[h] = acc
                        # Save best layer state for head h
                        best_state[h] = {k: v.detach().cpu().clone() for k, v in model.heads[h].state_dict().items()}
                    if wandb is not None:
                        wandb.log({f"{head_name_prefix}{h}/val_acc_epoch": acc, f"{head_name_prefix}{h}/epoch": ep})

        if scheduler is not None:
            scheduler.step()

        avg_loss = (total_loss / total_seen) if total_seen > 0 else 0.0
        post = {"loss": f"{avg_loss:.4f}"}
        if n_val > 0:
            # Show mean best val acc so far
            vals = [v for v in best_val.values() if v > float('-inf')]
            if len(vals) > 0:
                post["val"] = f"{(sum(vals)/len(vals)):.4f}"
        pbar.set_postfix(post)

    # Restore best head states if we tracked validation
    if n_val > 0:
        for h in heads:
            if best_state[h] is not None:
                model.heads[h].load_state_dict(best_state[h])

    # Final train/val metrics and export weights
    model.eval()
    out = {}
    with torch.no_grad():
        logits_train = model(X_train)
        logits_val = model(X_val) if n_val > 0 else None
        for h in heads:
            train_acc = accuracy(logits_train[h], head_y_train[h])
            val_acc = accuracy(logits_val[h], head_y_val[h]) if n_val > 0 else None
            state = {k: v.detach().cpu().numpy() for k, v in model.heads[h].state_dict().items()}
            out[h] = {
                "train_acc": float(train_acc),
                "val_acc": (float(val_acc) if val_acc is not None else None),
                "weights": state.get("weight"),
                "bias": state.get("bias"),
            }
    return out

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train linear probes per concept head on image embeddings (CE+L2)."
        )
    )
    parser.add_argument("--emb-path", type=str, default="/mnt/lustre/work/oh/owl661/complin_code/complinearity/outputs/clip_models_laion/clean_dsprites/clean_dsprites_embeddings_cat_cat_pixels.pkl", help="Path to embeddings PKL/NPZ")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save probes (default: <emb_dir>/probes)")
    parser.add_argument(
        "--heads",
        type=str,
        default="auto",
        help="Ignored. Heads are fixed from meta['dataset']. Kept only for CLI compatibility.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0, help="L2 on weights (Adam weight_decay)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.95, help="Fraction of data used for validation")
    parser.add_argument("--no-cosine-decay", dest="cosine_decay", action="store_false", help="Disable cosine annealing LR schedule (Adam only)")
    parser.add_argument("--eta-min", type=float, default=0.0, help="Minimum LR for cosine decay")

    args = parser.parse_args()
    if not hasattr(args, "cosine_decay"):
        setattr(args, "cosine_decay", True)
    payload = load_embeddings(args.emb_path)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Missing or invalid 'meta' dict in embeddings payload.")
    if "image_embeds" not in payload:
        raise ValueError("Missing 'image_embeds' in embeddings payload.")
    X_img = np.asarray(payload["image_embeds"])  # (N,D)
    emb_source = {"type": "image_embeds"}
    random_init = bool(meta.get("random_init", False))
    random_seed = meta.get("random_seed", None)
    out_json, out_prefix = get_output_paths(
        args.emb_path,
        args.out_dir,
        args.lr,
        random_init=random_init,
        random_seed=random_seed,
    )
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            existing = json.load(f)
        cfg = existing.get("train_cfg", {}) if isinstance(existing, dict) else {}
        if cfg.get("seed") == args.seed and cfg.get("lr") == args.lr:
            print(f"[skip] existing probes JSON matches seed/lr: {out_json}")
            return

    # Heads are fixed by dataset protocol.
    dataset_label = str(meta.get("dataset", "")).strip().lower()
    heads_by_dataset = {
        "dsprites": ["shape", "size", "orientation", "posx", "posy", "color"],
        "clean_dsprites": ["color", "shape", "size", "orientation", "posx", "posy"],
        "mpi3d": [
            "object-color",
            "object-shape",
            "object-size",
            "camera-height",
            "background-color",
            "horizontal-axis",
            "vertical-axis",
        ],
        "pug": ["character", "world", "size", "texture"],
    }
    if dataset_label not in heads_by_dataset:
        raise ValueError(
            f"Unsupported dataset '{dataset_label}' in embeddings meta. "
            f"Expected one of: {sorted(heads_by_dataset.keys())}"
        )
    heads = heads_by_dataset[dataset_label]
    results = {
        "emb_path": args.emb_path,
        "heads": heads,
        "train_cfg": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "val_split": args.val_split,
            "cosine_decay": args.cosine_decay,
            "eta_min": args.eta_min,
            "embedding_source": emb_source,
        },
        "probes": {},
        "embedding_source": emb_source,
    }

    wb = None
    if wandb is not None:
        base = os.path.basename(args.emb_path)
        ds_tag = str(meta.get("dataset", "unknown")).strip().lower() or "unknown"
        val_tag = str(args.val_split).replace(".", "p")
        run_name = f"probes_{ds_tag}_val{val_tag}_{os.path.splitext(base)[0]}"
        cfg = {
            "emb_path": args.emb_path,
            "heads": heads,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "val_split": args.val_split,
            "meta": meta,
            "cosine_decay": args.cosine_decay,
            "eta_min": args.eta_min,
            "embedding_source": emb_source,
        }
        project = os.environ.get("WANDB_PROJECT", "public-complinearity-probes")
        entity = os.environ.get("WANDB_ENTITY", None)
        wb = wandb.init(project=project, entity=entity, name=run_name, config=cfg, reinit=True)

    # Build labels/values map for joint training
    head_to_y = {}
    head_to_values = {}
    y_map = payload.get("y") if isinstance(payload.get("y"), dict) else None
    v_map = payload.get("values") if isinstance(payload.get("values"), dict) else None
    for head in heads:
        y = None
        values = None
        if y_map is not None and head in y_map:
            y = np.asarray(y_map[head]).astype(np.int64)
            if v_map is not None and head in v_map and len(v_map[head]) > 0:
                values = list(v_map[head])
        else:
            y_key = f"y_{head}"
            v_key = f"values_{head}"
            if y_key in payload and v_key in payload:
                y = np.asarray(payload[y_key]).astype(np.int64)
                values = list(payload[v_key])
        if y is None:
            raise ValueError(f"Missing labels for head '{head}' in payload.")
        if values is None or len(values) == 0:
            uniq = np.unique(y)
            values = [int(v) for v in uniq.tolist()]
        head_to_y[head] = y
        head_to_values[head] = values
        if wb is not None:
            wandb.log({f"{head}/num_classes": len(values)})
        results["probes"][head] = {"num_classes": len(values)}

    # Guard: ensure we have at least one valid head
    if len(head_to_y) == 0:
        raise ValueError("No valid heads found for probing.")

    # Jointly train all heads
    joint_out = train_all_heads_joint(
        X=X_img,
        head_to_y=head_to_y,
        head_to_values=head_to_values,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
        log_every=max(1, args.epochs // 10),
        head_name_prefix="",
        cosine_decay=args.cosine_decay,
        eta_min=args.eta_min,
    )

    # Save per-head weights and finalize results
    for head in heads:
        if head not in joint_out:
            continue
        po = joint_out[head]
        head_npz = f"{out_prefix}_{head}.npz"
        np.savez_compressed(
            head_npz,
            W=po["weights"],
            b=po["bias"],
            values=np.array(head_to_values[head], dtype=object),
            head=np.array([head], dtype=object),
        )
        if wb is not None:
            wandb.log({f"{head}/probe_top1": po["train_acc"], f"{head}/val_top1": (po["val_acc"] if po["val_acc"] is not None else None)})
        results["probes"][head].update({
            "train_top1": float(po["train_acc"]),
            "val_top1": (float(po["val_acc"]) if po["val_acc"] is not None else None),
            "weights_path": head_npz,
            "epochs": args.epochs,
        })

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_json}")
    if wb is not None:
        wandb.config.update({"probes_json": out_json}, allow_val_change=True)
        wandb.summary["probes_json"] = out_json
        wandb.summary["emb_path"] = args.emb_path
        wandb.finish()


if __name__ == "__main__":
    main()
