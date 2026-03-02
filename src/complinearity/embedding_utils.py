import os

import torch

import torchvision.transforms as T



def randomly_reinitialize_clip_model(model, std=0.02, seed=None):
    """Randomly reinitialize parameters of an OpenAI CLIP model.

    Applies reset_parameters where available and falls back to normal_(0,std) / zeros_.
    """
    if seed is not None:
        torch.manual_seed(int(seed))

    def reset_module(m):
        import torch.nn as nn
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.reset_parameters()
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight, mean=0.0, std=std)
        elif isinstance(m, nn.LayerNorm):
            if getattr(m, "elementwise_affine", False):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    model.apply(reset_module)

    # Secondary pass: be conservative. Only zero true biases; keep LayerNorm weights as ones;
    # lightly reinit multi-d weights. Leave other 1D scalars (e.g., logit_scale) as-is.
    for name, param in model.named_parameters(recurse=True):
        if not isinstance(param, torch.nn.Parameter):
            continue
        if name.endswith("bias"):
            torch.nn.init.zeros_(param)
        elif ("ln" in name) and name.endswith("weight"):
            # Preserve LayerNorm gamma to avoid collapsing activations
            torch.nn.init.ones_(param)
        elif param.dim() >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=std)
        else:
            # Leave other 1D scalars alone
            pass


def load_backend(
    backend,
    model_name,
    pretrained,
    device,
    random_init=False,
    random_seed=None,
    random_std=0.02,
):
    """Load CLIP/OpenCLIP/DINO backend. Returns (model, preprocess, used_pretrained_tag)."""
    backend = str(backend).strip().lower()

    if backend == "cat":
        # "cat" = concatenate raw pixels into one flat vector per image.
        # No model/preprocess required.
        return None, None, "pixels"

    if backend == "clip":
        import clip
        model, preprocess = clip.load(model_name, device=device, jit=False)

        if bool(random_init):
            randomly_reinitialize_clip_model(model, std=float(random_std), seed=random_seed)

        # OpenAI CLIP models are always pretrained; tweak tag when random init is used
        used_tag = "openai_rand" if bool(random_init) else "openai"
        if bool(random_init) and (random_seed is not None):
            used_tag = f"openai_rand_s{int(random_seed)}"
        return model, preprocess, used_tag

    if backend == "openclip":
        import open_clip

        # Map SigLIP/SigLIP2 HF ids to native OpenCLIP names when needed
        name_lc = (model_name or "").lower()
        resolved_model = None
        resolved_pretrained = None
        siglip_map = {
            "hf-hub:google/siglip-base-patch16-224": ("ViT-B-16-SigLIP", "webli"),
            "hf-hub:google/siglip-base-patch16-256": ("ViT-B-16-SigLIP-256", "webli"),
            "hf-hub:google/siglip-large-patch16-256": ("ViT-L-16-SigLIP-256", "webli"),
            "hf-hub:google/siglip2-base-patch16-224": ("ViT-B-16-SigLIP2", "webli"),
            "hf-hub:google/siglip2-base-patch16-256": ("ViT-B-16-SigLIP2-256", "webli"),
            "hf-hub:google/siglip2-large-patch16-256": ("ViT-L-16-SigLIP2-256", "webli"),
            "hf-hub:google/siglip2-large-patch16-384": ("ViT-L-16-SigLIP2-384", "webli"),
            "hf-hub:google/siglip2-so400m-patch16-256": ("ViT-SO400M-16-SigLIP2-256", "webli"),
        }
        for key, (m_id, pt_id) in siglip_map.items():
            if name_lc.startswith(key):
                resolved_model, resolved_pretrained = m_id, pt_id
                break

        if resolved_model is None:
            # Standard OpenCLIP path
            pretrained_tag = pretrained or "laion2b_s32b_b79k"
            oc_pretrained = None if bool(random_init) else pretrained_tag
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=oc_pretrained, device=device
            )
            if bool(random_init):
                randomly_reinitialize_clip_model(model, std=float(random_std), seed=random_seed)
                used_tag = f"{pretrained_tag}_rand"
                if random_seed is not None:
                    used_tag = f"{pretrained_tag}_rand_s{int(random_seed)}"
            else:
                used_tag = pretrained_tag
        else:
            oc_pretrained = None if bool(random_init) else resolved_pretrained
            model, _, preprocess = open_clip.create_model_and_transforms(
                resolved_model, pretrained=oc_pretrained, device=device
            )
            if bool(random_init):
                randomly_reinitialize_clip_model(model, std=float(random_std), seed=random_seed)
                base_tag = resolved_pretrained or "webli"
                used_tag = f"{base_tag}_rand"
                if random_seed is not None:
                    used_tag = f"{base_tag}_rand_s{int(random_seed)}"
            else:
                used_tag = resolved_pretrained

        return model, preprocess, used_tag

    if backend in {"dino", "timm"}:
        import timm
        from timm.data import create_transform, resolve_model_data_config


        class DINOVisualWrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def encode_image(self, x):
                feats = self.inner.forward_features(x)
                head_input = feats
                head = getattr(self.inner, "forward_head", None)
                if callable(head):
                    embedding = head(head_input, pre_logits=True)
                else:
                    if isinstance(feats, dict):
                        first_val = next(iter(feats.values()))
                        if isinstance(first_val, torch.Tensor):
                            embedding = first_val[:, 0, ...] if first_val.ndim >= 2 else first_val
                        else:
                            raise RuntimeError("DINO forward_features returned dict without tensor values.")
                    elif isinstance(feats, torch.Tensor):
                        embedding = feats[:, 0, ...] if feats.ndim >= 2 else feats
                    else:
                        raise RuntimeError(f"Unexpected forward_features output type: {type(feats)}")
                if isinstance(embedding, (tuple, list)):
                    embedding = embedding[0]
                return embedding

            def forward(self, x):
                return self.encode_image(x)

        used_tag = "timm_default"
        load_pretrained = True
        pretrained_cfg = None
        state_dict_path = None
        if pretrained is not None:
            lowered = str(pretrained).strip().lower()
            if os.path.isfile(str(pretrained)):
                state_dict_path = str(pretrained)
                load_pretrained = False
                used_tag = f"state_dict:{os.path.basename(state_dict_path)}"
            elif lowered in {"false", "0", "no", "off", "none", "random"}:
                load_pretrained = False
                used_tag = "random"
            elif lowered in {"default", "true", "timm"} or lowered == "laion400m_e32":
                # keep defaults
                pass
            else:
                pretrained_cfg = pretrained
                used_tag = str(pretrained)

        model = timm.create_model(
            model_name,
            pretrained=load_pretrained,
            pretrained_cfg=pretrained_cfg,
        )
        if state_dict_path is not None:
            checkpoint = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            used_tag = used_tag + "+loaded"

        model.to(device)
        model.eval()
        preprocess_cfg = resolve_model_data_config(model)
        preprocess = create_transform(**preprocess_cfg)
        wrapped = DINOVisualWrapper(model).to(device)
        return wrapped, preprocess, used_tag

    # No model requested; return None and default preprocess
    return None, None, None


def model_signature(backend, model_name, pretrained_tag):
    pt = pretrained_tag or "openai"
    return f"{backend}:{model_name}:{pt}"


def sanitize_filename_component(value):
    return value.replace(os.sep, "-").replace(" ", "_")


def build_output_path(
    output_path,
    backend,
    model_name,
    pretrained_tag=None,
    dataset_label="pug",
):
    if output_path.endswith(".npz"):
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return output_path
    base_dir_root = output_path if output_path else "."
    base_dir = os.path.join(base_dir_root, dataset_label)
    os.makedirs(base_dir, exist_ok=True)
    pt = pretrained_tag or "openai"
    fname = f"{sanitize_filename_component(dataset_label)}_embeddings_{sanitize_filename_component(backend)}_{sanitize_filename_component(model_name)}_{sanitize_filename_component(pt)}.npz"
    return os.path.join(base_dir, fname)

def sibling_with_ext(path_like, new_ext):
    base, _ = os.path.splitext(path_like)
    return base + new_ext


def normalize_pretrained_input(pretrained, backend):
    if str(backend).strip().lower() == "cat":
        return None
    if pretrained is None:
        return None
    lowered = str(pretrained).strip().lower()
    if lowered in {"true", "1", "yes", "on"}:
        # For openai clip this value is ignored; for openclip we'll use default tag later
        return None
    if lowered in {"false", "0", "no", "off"}:
        return None
    return pretrained
