import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb


def login_wandb():
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key


login_wandb()
def load_embeddings(embeddings_path):
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(embeddings_path)
    ext = os.path.splitext(embeddings_path)[1].lower()

    if ext == ".pkl":
        with open(embeddings_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise ValueError("Pickle payload must be a dict")
        return payload

    if ext == ".npz":
        data = np.load(embeddings_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    raise ValueError("Unsupported format; use .pkl or .npz")

def select_embedding_matrix(
    payload,
):
    X = np.array(payload["image_embeds"], dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    emb_source = {"type": "image_embeds", "sample_count": int(X.shape[0]), "hidden_dim": int(X.shape[1])}

    return X, None, emb_source

def get_probes_tag(embeddings_path):
    base = os.path.basename(embeddings_path)
    base_noext = os.path.splitext(base)[0]
    return base_noext.replace("pug_embeddings_", "pug_probes_") if base_noext.startswith("pug_embeddings_") else f"pug_probes_{base_noext}"

def load_probes_json(
    probes_json_path,
):
    with open(probes_json_path, "r") as f:
        return json.load(f), probes_json_path

def load_head_weights(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    data = np.load(weights_path, allow_pickle=True)
    # Expect keys: W, b, values, head
    loaded = {}
    for key in data.files:
        loaded[key] = data[key]
    return loaded

def load_embeddings_and_probes(
    embeddings_path,
    probes_json_path,
):
    emb_payload = load_embeddings(embeddings_path)
    probes_json, probes_json_path = load_probes_json(probes_json_path)
    head_to_weights = {}
    probes = probes_json.get("probes", {})
    for head_name, meta in probes.items():
        weights_path = meta.get("weights_path")
        if isinstance(weights_path, str) and os.path.exists(weights_path):
            head_to_weights[head_name] = load_head_weights(weights_path)
        else:
            # Try constructing from prefix if weights_path missing
            prefix = os.path.splitext(probes_json_path)[0]
            candidate = f"{prefix}_{head_name}.npz"
            if os.path.exists(candidate):
                head_to_weights[head_name] = load_head_weights(candidate)
    return emb_payload, probes_json, head_to_weights, probes_json_path


def infer_heads_from_payload(payload):
    heads = []
    # PKL-style
    values_map = payload.get("values")
    if isinstance(values_map, dict) and len(values_map) > 0:
        heads = list(values_map.keys())
    else:
        # NPZ-style: values_<head>
        for k in payload.keys():
            if isinstance(k, str) and k.startswith("values_"):
                heads.append(k[len("values_"):])
    return sorted(list(dict.fromkeys(heads)))


def get_y_and_values_for_head(payload, head):
    if isinstance(payload.get("y"), dict) and isinstance(payload.get("values"), dict):
        y_map = payload["y"]
        values_map = payload["values"]
        if head in y_map and head in values_map:
            y = np.asarray(y_map[head]).astype(np.int64)
            values_list = list(values_map[head])
            return y, values_list
    y_key = f"y_{head}"
    values_key = f"values_{head}"
    if y_key in payload and values_key in payload:
        y = np.asarray(payload[y_key]).astype(np.int64)
        values_list = list(payload[values_key])
        return y, values_list
    raise KeyError(f"Missing labels/values for head '{head}'")


def compute_concept_value_means(
    payload,
    heads=None,
    center_per_concept=True,
):
    if "image_embeds" not in payload:
        raise ValueError("Embeddings payload missing 'image_embeds'")
    X = np.asarray(payload["image_embeds"], dtype=np.float32)  # (N, d)
    return compute_concept_value_means_from_matrix(X, payload, heads=heads, center_per_concept=center_per_concept)


def get_r2_per_concept_probe_span(
    X_use,
    payload,
    centered_vocab,
    head_to_weights,
):
    heads = infer_heads_from_payload(payload)
    
    r2_by_head = {}
    for head in heads:
        weights = head_to_weights[head]
        W = weights["W"]
        active_mask = centered_vocab[head].get("counts") > 0
        W_take = W[active_mask]
        ys_head = get_y_and_values_for_head(payload, head)
        
        U, S, Vt = np.linalg.svd(W_take, full_matrices=False)
        r = int(np.sum(S > 1e-12))
        B = Vt[:r, :].T # (d, r)
        
        P_proj = B @ B.T # (d, d)
        
        X_on_probe_span = X_use @ P_proj # (N, d)
        
        centered_means = centered_vocab[head]["means"]
        centered_means_on_probe_span = centered_means @ P_proj # (C, d)
        
        approx_points = np.zeros_like(X_on_probe_span)

        num_max_classes = int(ys_head[0].max()) + 1
        
        for i in range(num_max_classes):
            idx = np.where(ys_head[0] == i)[0]
            if idx.size == 0:
                # print(f"No samples for class {i}")
                continue
            centered_means_on_probe_span_c = centered_means_on_probe_span[i]
            approx_points[idx] = centered_means_on_probe_span_c
            # approx_points[idx] = X_on_probe_span[idx].mean(axis=0)
        resid = X_on_probe_span - approx_points
        sse = np.sum(resid * resid)
        sst = np.sum((X_on_probe_span - X_on_probe_span.mean(axis=0, keepdims=True)) ** 2)
        r2 = 1 - sse / (sst + 1e-9)
        r2_by_head[head] = float(r2)
        wb_log({f"heads_probe_span_r2/{head}": float(r2)})
        print(f"R2 for {head}: {r2}")
    if len(r2_by_head) > 0:
        avg_r2 = float(np.mean(list(r2_by_head.values())))
        wb_log({"heads_probe_span_r2/avg_over_heads": avg_r2})
    return r2_by_head
            
def compute_concept_value_means_from_matrix(
    X_use,
    payload,
    heads=None,
):
    if X_use.ndim != 2:
        raise ValueError("X must be 2D (N, d)")
    N, d = X_use.shape[0], X_use.shape[1]
    indices = None
    
    if heads is None:
        heads = infer_heads_from_payload(payload)
    raw = {}
    centered = {}

    for head in heads:
        y, values_list = get_y_and_values_for_head(payload, head)
        
        # Use labels as-is; allow holes in class ids for consistency with probes
        y = np.asarray(y, dtype=np.int64)
        num_classes = int(y.max()) + 1 if y.size > 0 else 0
        labels = list(values_list) if values_list is not None else list(range(num_classes))
        means = np.zeros((num_classes, d), dtype=np.float32)
        counts = np.zeros((num_classes,), dtype=np.int64)
        for cls in range(num_classes):
            idx = np.where(y == cls)[0]
            counts[cls] = int(idx.size)
            if idx.size > 0:
                means[cls] = np.mean(X_use[idx], axis=0)
        raw[head] = {"labels": labels, "means": means, "counts": counts}

        centered[head] = {"labels": list(raw[head]["labels"]), "means": means.copy(), "counts": counts}

    return raw, centered
def k_at_threshold(evr, threshold=0.95):
    evr = np.asarray(evr, dtype=float)
    if evr.size == 0:
        return 0
    cev = np.cumsum(evr)
    idx = int(np.searchsorted(cev, threshold, side="left"))
    return min(idx + 1, evr.size)

def wb_log(data):
    wandb.log(data)

def norm_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / (n + 1e-9)
def compute_means_metrics_and_orthogonality(centered_vocab, out_dir, prefix):
    metrics = {"k95_means": {}, "pairwise_mean_abs_cos": {}, "avg_same_concept_abs_cos": None, "avg_cross_concept_abs_cos": None}
    # dimensionalities
    for head, info in centered_vocab.items():
        means_h = np.array(info.get("means"))
        count_nonzero_mask = info.get("counts") > 0
        means_h = means_h[count_nonzero_mask]
        _, S_h, _ = np.linalg.svd(means_h, full_matrices=False)

        var = S_h ** 2
        total = float(var.sum()) if var.size > 0 else 0.0

        evr = var / total

        k95 = k_at_threshold(evr, 0.95)
        metrics["k95_means"][head] = k95
        wb_log({f"{prefix}/{head}/k95_means": k95})

    # orthogonality
    heads = list(centered_vocab.keys())
    same_pool = []
    diff_pool = []
    for i, ha in enumerate(heads):
        ma = centered_vocab[ha]["means"]
        count_nonzero_mask_a = centered_vocab[ha]["counts"] > 0
        ma = ma[count_nonzero_mask_a]
        for j, hb in enumerate(heads[i:]):
            hb = heads[i + j]
            mb = centered_vocab[hb]["means"]
            count_nonzero_mask_b = centered_vocab[hb]["counts"] > 0
            mb = mb[count_nonzero_mask_b]
            
            sims = np.abs(norm_rows(ma) @ norm_rows(mb).T)
            if ha == hb:
                mask = ~np.eye(sims.shape[0], dtype=bool)
                same_vals = sims[mask]
                same_pool.extend([float(x) for x in same_vals.ravel()])
                mean_abs_cos = float(np.mean(same_vals)) 
            else:
                diff_pool.extend([float(x) for x in sims.ravel()])
                mean_abs_cos = float(np.mean(sims)) 

            key = f"{ha}_vs_{hb}"
            metrics["pairwise_mean_abs_cos"][key] = mean_abs_cos

            wb_log({f"{prefix}/means_orthogonality/{key}": mean_abs_cos})
           
    metrics["avg_same_concept_abs_cos"] = float(np.mean(same_pool))
    metrics["avg_cross_concept_abs_cos"] = float(np.mean(diff_pool)) 
    wb_log({f"{prefix}/means_orthogonality/avg_same_concept_abs_cos": metrics["avg_same_concept_abs_cos"]})
    wb_log({f"{prefix}/means_orthogonality/avg_cross_concept_abs_cos": metrics["avg_cross_concept_abs_cos"]})
    return metrics


def compute_linear_factorization_r2_from_matrix(
    Xc,
    payload,
    centered_vocab,
    heads=None,
):
    N, d = Xc.shape
    if heads is None:
        heads = list(centered_vocab.keys())
    # Build reconstruction by summing centered means for each head using label
    X_hat = np.zeros_like(Xc)
    for h in heads:
        means_h = np.asarray(centered_vocab[h].get("means"))  # (C,d)
        y_h= np.asarray(get_y_and_values_for_head(payload, h)[0]).astype(np.int64)
        
        X_hat += means_h[y_h]

    resid = Xc - X_hat
    sse = float(np.sum(resid * resid))
    sst = float(np.sum(Xc * Xc)) + 1e-9
    r2 = float(1.0 - sse / sst) 

    eps = 1e-9
    U_svd, S_svd, Vt_svd = np.linalg.svd(Xc, full_matrices=False)
    denom = float(N - 1) if N > 1 else float(max(N, 1))
    eigvals = (S_svd ** 2) / denom
    inv_sqrt = 1.0 / np.sqrt(eigvals + eps)
    X_tmp = Xc @ Vt_svd.T
    X_tmp *= inv_sqrt
    Xc_w = X_tmp @ Vt_svd
    X_hat_w = np.zeros_like(Xc_w)
    for h in heads:
        means_h = np.asarray(centered_vocab[h].get("means"))  # (C,d)
        y_h = np.asarray(get_y_and_values_for_head(payload, h)[0]).astype(np.int64)
        M_tmp = means_h @ Vt_svd.T
        M_tmp *= inv_sqrt
        means_h_w = M_tmp @ Vt_svd
        X_hat_w += means_h_w[y_h]
    resid_w = Xc_w - X_hat_w
    sse_w = float(np.sum(resid_w * resid_w))
    sst_w = float(np.sum(Xc_w * Xc_w))
    r2_w = float(1.0 - sse_w / sst_w) if sst_w > 0 else float("nan")

    return {"r2": r2, "sse": sse, "sst": sst, "r2_whitened": r2_w, "sse_whitened": sse_w, "sst_whitened": sst_w}

def main():
    parser = argparse.ArgumentParser(description="Load embeddings and their trained probes for analysis")
    parser.add_argument("--emb-path", type=str, help="Path to embeddings .pkl/.npz", default="/mnt/lustre/work/oh/owl661/complin_code/complinearity/outputs/clip_models_laion/clean_dsprites/clean_dsprites_embeddings_pixels.pkl")
    parser.add_argument("--probes-json", type=str, default="/mnt/lustre/work/oh/owl661/complin_code/complinearity/outputs/clip_models_laion/clean_dsprites/_probes/_probes_clean_dsprites_embeddings_lr0p001.json")
    args = parser.parse_args()

    # Initialize Weights & Biases run
    run_name = f"analysis_{os.path.splitext(os.path.basename(args.emb_path))[0]}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "public3-complinearity-analysis"),
        name=run_name,
        reinit=True,
        config={"emb_path": args.emb_path},
    )

    emb_payload, probes_json, head_to_weights, probes_json_path = load_embeddings_and_probes(
        args.emb_path,
        args.probes_json,
    )

    selected_X, _, emb_source = select_embedding_matrix(emb_payload)
    
    # Compute per-(concept,value) mean embeddings (pre-projection)
    raw_vocab, centered_vocab = compute_concept_value_means_from_matrix(
        selected_X,
        emb_payload,
    )
    
    # ============= per-concept stats =============
    get_r2_per_concept_probe_span(selected_X, emb_payload, centered_vocab, head_to_weights)
    
    # =============================================
    
    
    # Prepare plots directory (timestamped) for all figures in this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(os.path.dirname(args.emb_path), "probes", "analysis_plots", timestamp)
    raw_dir = os.path.join(base_dir, "raw")
    projected_dir = os.path.join(base_dir, "projected")
    probes_dir = os.path.join(base_dir, "probes")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(projected_dir, exist_ok=True)
    os.makedirs(probes_dir, exist_ok=True)

    # Model summary/meta reporting
    emb = np.asarray(selected_X, dtype=np.float32)
    emb_dim = int(emb.shape[1]) if emb.ndim == 2 else None
    num_images = int(emb.shape[0]) if emb.ndim == 2 else -1
    meta = emb_payload.get("meta", {}) if isinstance(emb_payload.get("meta", {}), dict) else {}
    model_name = meta.get("model_name") or meta.get("model") or meta.get("clip_variant") or meta.get("backbone")
    backend = meta.get("backend") or meta.get("framework") or meta.get("provider")
    pretrained = meta.get("pretrained")
    caption_types = meta.get("caption_types")
    dataset = meta.get("dataset")
    if not dataset:
        base_noext = os.path.splitext(os.path.basename(args.emb_path))[0]
        if "_embeddings_" in base_noext:
            dataset = base_noext.split("_embeddings_", 1)[0]
    model_sig = meta.get("model_signature")
    train_cfg = probes_json.get("train_cfg", {}) if isinstance(probes_json.get("train_cfg", {}), dict) else {}
    val_split = train_cfg.get("val_split")
    lr = train_cfg.get("lr")
    model_info = {
        "emb_path": args.emb_path,
        "probes_json": probes_json_path,
        "emb_dim": emb_dim,
        "num_images": num_images,
        "model_name": model_name,
        "backend": backend,
        "pretrained": pretrained,
        "caption_types": caption_types,
        "dataset": dataset,
        "model_signature": model_sig,
        "meta": meta,
        "embedding_source": emb_source,
        "val_split": val_split,
        "lr": lr,
    }
    wandb.config.update({
        "dataset": dataset or "",
        "backend": backend or "",
        "model_name": model_name or "",
        "pretrained": pretrained or "",
        "caption_types": caption_types or [],
        "num_samples": num_images,
        "emb_dim": emb_dim if emb_dim is not None else -1,
        "emb_path": args.emb_path,
        "probes_json": probes_json_path,
            "model_signature": model_sig or "",
            "embedding_source": emb_source,
            "val_split": val_split,
            "lr": lr,
        }, allow_val_change=True)
    run = wandb.run
    if run is not None and dataset and model_name:
        new_name = f"analysis_{dataset}_{model_name}" + (f"_{pretrained}" if pretrained else "")
        run.name = new_name
    # Log to W&B
    wb_log({
        "model/name": model_name if model_name is not None else "",
        "model/backend": backend if backend is not None else "",
        "emb/dim": emb_dim if emb_dim is not None else -1,
        "emb/num_images": num_images,
        "emb/path": args.emb_path,
        "probes/json_path": model_info["probes_json"],
        "probes/val_split": val_split if val_split is not None else "",
        "probes/lr": lr if lr is not None else "",
    })
    print(f"Model: name={model_name} backend={backend} emb_dim={emb_dim} images={num_images}")

    # Plot SVD explained variance + orthogonality (RAW)
    raw_metrics = compute_means_metrics_and_orthogonality(centered_vocab, raw_dir, "raw")
  
    with open(os.path.join(raw_dir, "metrics.json"), "w") as f:
        json.dump(raw_metrics, f, indent=2)

    num_images = int(selected_X.shape[0])
    heads = sorted(list((probes_json.get("probes") or {}).keys()))
    print(f"Embeddings: {args.emb_path}")
    print(f"Images: {num_images}")
    print(f"Probes JSON: {probes_json_path}")
    print(f"Heads in JSON: {', '.join(heads) if heads else '(none)'}")
    print("Loaded weights:")
    for h in heads:
        loaded = head_to_weights.get(h)
        if loaded is None:
            print(f"  - {h}: MISSING weights")
        else:
            W = np.asarray(loaded.get("W"))
            b = np.asarray(loaded.get("b"))
            print(f"  - {h}: W {tuple(W.shape)}  b {tuple(b.shape)}")

    # Probes performance metrics (zero-shot and probe training accuracy)
    probes_metrics = {"probe_train_top1": {}, "probe_val_top1": {}, "num_classes": {}, "avg_probe_train_top1": None, "avg_probe_val_top1": None}
    pj_probes = (probes_json.get("probes") or {})
    for h in heads:
        meta = pj_probes.get(h, {})
        tr = meta.get("train_top1")
        nc = meta.get("num_classes")
        probes_metrics["probe_train_top1"][h] = float(tr)
        wb_log({f"probes/{h}/probe_train_top1": float(tr)})
        probes_metrics["probe_val_top1"][h] = float(meta.get("val_top1"))
        wb_log({f"probes/{h}/probe_val_top1": float(meta.get("val_top1"))})
        probes_metrics["num_classes"][h] = int(nc)
        wb_log({f"probes/{h}/num_classes": int(nc)})
    # Aggregates across heads
    def avg(d):
        vals = [float(v) for v in d.values()]
        return float(np.mean(vals))
    probes_metrics["avg_probe_train_top1"] = avg(probes_metrics["probe_train_top1"])
    probes_metrics["avg_probe_val_top1"] = avg(probes_metrics["probe_val_top1"])
    wb_log({"probes/avg_probe_train_top1": probes_metrics["avg_probe_train_top1"]})
    wb_log({"probes/avg_probe_val_top1": probes_metrics["avg_probe_val_top1"]})
    with open(os.path.join(probes_dir, "metrics.json"), "w") as f:
        json.dump(probes_metrics, f, indent=2)

    # Collect all class weight vectors across heads
    rows = []
    # labels = []
    for h in heads:
        loaded = head_to_weights.get(h)
        if loaded is None:
            continue
        W_h = np.asarray(loaded.get("W"))  # (C, d)
        active_mask = centered_vocab[h].get("counts") > 0
        W_take = W_h[active_mask]
        for i in range(W_take.shape[0]):
            rows.append(W_take[i])
    if len(rows) == 0:
        return
    W_total = np.stack(rows, axis=0)  # (m, d)
    # Orthonormal basis via SVD of W_total
    U, S, Vt = np.linalg.svd(W_total, full_matrices=False)
    # Rank threshold
    r = int(np.sum(S > 1e-6))
    B = Vt[:r, :].T  # (d, r) orthonormal basis for probe subspace
    # Project embeddings
    X = selected_X - selected_X.mean(axis=0, keepdims=True)
    Z = X @ B              # (N, r) coordinates in probe subspace
    X_proj = Z @ B.T       # (N, d) component explained by probes
    print(f"Coordinates shape: {Z.shape}")
   

    # ===== Plotting (probes) =====

    # Compute linear factorization R^2 for RAW
    lf_raw = compute_linear_factorization_r2_from_matrix(X, emb_payload, centered_vocab)
    wb_log({
        "raw/linear_factorization/r2": lf_raw["r2"],
        "raw/linear_factorization/sse": lf_raw["sse"],
        "raw/linear_factorization/sst": lf_raw["sst"],
        "raw/linear_factorization/r2_whitened": lf_raw.get("r2_whitened"),
        "raw/linear_factorization/sse_whitened": lf_raw.get("sse_whitened"),
        "raw/linear_factorization/sst_whitened": lf_raw.get("sst_whitened"),
    })

    # ===== Projected embeddings analysis (X_proj) =====
    # Recompute per-(concept,value) means on the component explained by probes
    raw_proj, centered_proj = compute_concept_value_means_from_matrix(
        X_proj,
        emb_payload,
    )
    # Compute linear factorization R^2 for PROJECTED 
    lf_proj = compute_linear_factorization_r2_from_matrix(
        X_proj,
        emb_payload,
        centered_proj,
        heads=list(centered_proj.keys()),
    )
    wb_log({
        "projected/linear_factorization/r2": lf_proj["r2"],
        "projected/linear_factorization/sse": lf_proj["sse"],
        "projected/linear_factorization/sst": lf_proj["sst"],
        "projected/linear_factorization/r2_whitened": lf_proj.get("r2_whitened"),
        "projected/linear_factorization/sse_whitened": lf_proj.get("sse_whitened"),
        "projected/linear_factorization/sst_whitened": lf_proj.get("sst_whitened"),
    })
    # ----------
    proj_metrics = compute_means_metrics_and_orthogonality(centered_proj, projected_dir, "projected")
    
    with open(os.path.join(projected_dir, "metrics.json"), "w") as f:
        json.dump(proj_metrics, f, indent=2)

if __name__ == "__main__":
    main()
