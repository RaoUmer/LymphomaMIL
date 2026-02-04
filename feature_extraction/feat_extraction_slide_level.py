import glob
import os
import gc
import h5py
import numpy as np
from tqdm import tqdm
import torch
import argparse

import encoders 

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True,
                    help='Path containing *.h5 files: <slide_id>_<label>.h5')
parser.add_argument('--out_path', type=str, required=True,
                    help='Where to save slide-level TITAN embeddings')
parser.add_argument('--encoder', type=str, default='titan',
                    choices=['titan'])
parser.add_argument('--tilesize', type=int, default=512)

# cap patches to avoid OOM
parser.add_argument('--max_patches', type=int, default=4096,
                    help='Max number of patches fed to TITAN per slide. '
                         'Attention is O(N^2); keep this modest.')
parser.add_argument('--sample_strategy', type=str, default='random',
                    choices=['random', 'grid'],
                    help='How to downsample patches if N > max_patches.')
parser.add_argument('--seed', type=int, default=0)

def load_conch_features(h5_path):
    """Load patch features + coords from a CONCH .h5 file."""
    with h5py.File(h5_path, 'r') as f:
        feats = f['features'][:]      # (N, D)
        coords = f['coords'][:]       # (N, 2)
    return feats, coords


def infer_patch_size(coords_np, fallback):
    coords_np = coords_np.astype(np.int64)
    xs = np.unique(coords_np[:, 0])
    ys = np.unique(coords_np[:, 1])

    if len(xs) > 1:
        dx = np.median(np.diff(np.sort(xs)))
    else:
        dx = fallback

    if len(ys) > 1:
        dy = np.median(np.diff(np.sort(ys)))
    else:
        dy = fallback

    return int(np.median([dx, dy]))


def downsample_patches(feats_np, coords_np, max_patches, strategy='random', seed=0):
    """Reduce patch count to max_patches to prevent TITAN OOM."""
    n = feats_np.shape[0]
    if n <= max_patches:
        return feats_np, coords_np

    rng = np.random.default_rng(seed)

    if strategy == 'random':
        idx = rng.choice(n, size=max_patches, replace=False)

    elif strategy == 'grid':
        # simple spatial grid subsampling:
        # bin coords into a coarse grid and sample evenly
        coords = coords_np.astype(np.int64)
        # normalize to 0..1 then grid
        x = coords[:, 0]
        y = coords[:, 1]
        x_norm = (x - x.min()) / (x.ptp() + 1e-8)
        y_norm = (y - y.min()) / (y.ptp() + 1e-8)

        g = int(np.sqrt(max_patches))
        gx = np.floor(x_norm * g).astype(int)
        gy = np.floor(y_norm * g).astype(int)
        grid_id = gx * g + gy

        # pick ~one per occupied cell, then fill randomly if still short
        uniq = np.unique(grid_id)
        idx_list = []
        for u in uniq:
            candidates = np.where(grid_id == u)[0]
            idx_list.append(rng.choice(candidates))
            if len(idx_list) >= max_patches:
                break
        idx = np.array(idx_list, dtype=int)
        if len(idx) < max_patches:
            remain = np.setdiff1d(np.arange(n), idx)
            extra = rng.choice(remain, size=max_patches - len(idx), replace=False)
            idx = np.concatenate([idx, extra])

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return feats_np[idx], coords_np[idx]

def compact_coords(coords_np, patch_size_lv0):
    """
    Reindex sparse coords onto a compact regular grid.
    Keeps relative layout, removes huge empty gaps.
    """
    coords_np = coords_np.astype(np.int64)

    xs = np.unique(coords_np[:, 0])
    ys = np.unique(coords_np[:, 1])

    x_map = {x:i for i, x in enumerate(np.sort(xs))}
    y_map = {y:i for i, y in enumerate(np.sort(ys))}

    x_idx = np.vectorize(x_map.get)(coords_np[:, 0])
    y_idx = np.vectorize(y_map.get)(coords_np[:, 1])

    coords_compact = np.stack([x_idx, y_idx], axis=1).astype(np.int64)
    coords_compact *= int(patch_size_lv0)
    return coords_compact


def extract_slide_embedding(conch_h5_path, out_file, model, device, args):
    feats_np, coords_np = load_conch_features(conch_h5_path)
    print("  original feats shape:", feats_np.shape, "coords shape:", coords_np.shape)

    coords_np = coords_np.astype(np.int64)
    patch_size_lv0 = infer_patch_size(coords_np, args.tilesize)
    print(f"  inferred patch_size_lv0: {patch_size_lv0}")

    def run_titan(feats_np_run, coords_np_run, dev):
        feats = torch.from_numpy(feats_np_run)
        coords = torch.from_numpy(coords_np_run)
        batch = {'features': feats, 'coords': coords}
        with torch.no_grad():
            out = model(batch, psize=patch_size_lv0, device=dev)
        # make sure tensors can be freed
        del batch, feats, coords
        return out

    # 1) try full patches with compact coords
    try:
        coords_compact = compact_coords(coords_np, patch_size_lv0)
        slide_emb = run_titan(feats_np, coords_compact, device)

    except torch.cuda.OutOfMemoryError:
        print("[OOM] Full TITAN failed. Clearing cache and retrying with sampling...")

        # hard cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) progressive retries with smaller caps
        caps = [args.max_patches, args.max_patches // 2, args.max_patches // 4]
        caps = [c for c in caps if c >= 512]

        slide_emb = None
        for cap in caps:
            try:
                feats_ds, coords_ds = downsample_patches(
                    feats_np, coords_np,
                    max_patches=cap,
                    strategy=args.sample_strategy,
                    seed=args.seed
                )
                coords_ds = compact_coords(coords_ds, patch_size_lv0)

                print(f"  retry cap={cap}: feats {feats_ds.shape}, coords {coords_ds.shape}")
                slide_emb = run_titan(feats_ds, coords_ds, device)
                break
            except torch.cuda.OutOfMemoryError:
                print(f"  [OOM] cap={cap} still too big -> shrinking")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # 3) last resort: CPU for this slide only
        if slide_emb is None:
            print("  [CPU FALLBACK] Running TITAN on CPU for this slide.")
            feats_ds, coords_ds = downsample_patches(
                feats_np, coords_np,
                max_patches=caps[-1] if caps else 1024,
                strategy=args.sample_strategy,
                seed=args.seed
            )
            coords_ds = compact_coords(coords_ds, patch_size_lv0)
            slide_emb = run_titan(feats_ds, coords_ds, torch.device("cpu"))

    # save
    slide_emb = slide_emb.detach().cpu().numpy()
    if slide_emb.ndim == 1:
        slide_emb = slide_emb[None, :]

    with h5py.File(out_file, "w") as hf:
        hf.create_dataset("features", data=slide_emb)

    print(f"[SAVED] {out_file}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _, _ = encoders.get_encoder(args.encoder)
    model.to(device).eval()

    pattern = os.path.join(args.data_path, "*.h5")
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} CONCH feature files.")

    out_dir = os.path.join(args.out_path, args.encoder)
    os.makedirs(out_dir, exist_ok=True)

    for h5_file in tqdm(files):
        fname = os.path.basename(h5_file)
        out_file = os.path.join(out_dir, fname)

        if os.path.exists(out_file):
            print(f"[SKIP] Already exists: {out_file}")
            continue

        slide_name = os.path.splitext(fname)[0]
        print(f"\nProcessing slide: {slide_name}")
        print(f"  Source file: {h5_file}")
        print(f"  Output file: {out_file}")

        extract_slide_embedding(
            conch_h5_path=h5_file,
            out_file=out_file,
            model=model,
            device=device,
            args=args
        )


if __name__ == "__main__":
    main()
