import glob
import os
import gc
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import resnet_custom
import pdb
import encoders
import argparse
import utils  

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', help='root path containing sqlite slides')
parser.add_argument('--out_path', type=str, default='', help='path to output data')
parser.add_argument('--encoder', type=str, default='',
                    choices=[
                        'tres50_imagenet',
                        'ctranspath',
                        'phikon',
                        'uni',
                        'uni2',
                        'gigapath',
                        'virchow',
                        'virchow2',
                        'h-optimus-1',
                        'h0-mini',
                        'dinosmall',
                        'dinobase',
                        'kaiko_vitl14',
                        'conchv1_5'
                    ],
                    help='choice of encoder')
parser.add_argument('--tilesize', type=int, default=224, help='tile size (passed to load_slide_data)')


def extract_Features(images,
                     coords,
                     slide_name,
                     label,
                     model,
                     transform,
                     ndim,
                     device,
                     args):
    """
    images: list of PIL.Image (already resized to tilesize in load_slide_data)
    coords: list of (x, y)
    label : slide-level diagnosis (string/int)
    """
    encoder_out_dir = os.path.join(args.out_path, args.encoder)
    os.makedirs(encoder_out_dir, exist_ok=True)

    #model.eval()
    feats_list = []

    with torch.no_grad():
        for img in images:
            # Ensure PIL.Image in RGB
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.convert('RGB')

            # Transform and forward
            img_t = transform(img)                  # (C, H, W)
            batch_t = img_t.unsqueeze(0).to(device) # (1, C, H, W)

            features = model.forward(batch_t)
            # If encoder returns dict (some custom models do)
            if isinstance(features, dict):
                # adjust key if needed depending on your encoder implementation
                features = features.get('feats', list(features.values())[0])

            features = features.cpu().detach().numpy()  # (1, ndim)
            feats_list.append(features)

    if len(feats_list) == 0:
        print(f'[WARNING] No patches for slide {slide_name}, skip.')
        return

    feats = np.concatenate(feats_list, axis=0).astype(np.float32)  # (N, ndim)
    coords_arr = np.array(coords)

    if coords_arr.shape[0] != feats.shape[0]:
        print(f'[WARNING] coords length ({coords_arr.shape[0]}) != '
              f'features length ({feats.shape[0]}) for slide {slide_name}')

    print(f'features size: {feats.shape}')

    out_file = os.path.join(encoder_out_dir, f'{slide_name}_{label}.h5')
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("features", data=feats, compression='gzip')
        f.create_dataset("coords", data=coords_arr, compression='gzip')
        # Store slide-level label once as an attribute
        f.attrs["slide_label"] = str(label)

    print(f'[SAVED] {out_file}')


def main():
    """
    Feature extraction directly from sqlite slides:

      - find all *.sqlite under args.data_path
      - for each sqlite:
          * get max level
          * load images, coords, diagnosis via utils.load_slide_data
          * extract features
          * save features, coords, labels to .h5
    """
    global args
    args = parser.parse_args()

    # Set up encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, ndim = encoders.get_encoder(args.encoder)
    model.model.to(device)

    # Ensure encoder-specific output directory exists
    encoder_out_path = os.path.join(args.out_path, args.encoder)
    os.makedirs(encoder_out_path, exist_ok=True)

    # Collect all sqlite slides recursively
    sqlite_files = glob.glob(os.path.join(args.data_path, '**', '*.sqlite'),
                             recursive=True)
    sqlite_files = sorted(sqlite_files)

    print(f'Found {len(sqlite_files)} sqlite slides under {args.data_path}')

    for sqlite_path in sqlite_files:
        slide_name = os.path.basename(os.path.dirname(sqlite_path))

        print(f'\n[INFO] Processing slide: {slide_name}')
        print(f'       SQLite path: {sqlite_path}')

        # Read level + data from sqlite
        max_level = utils.get_max_level(sqlite_path)
        if max_level is None:
            print(f'[WARNING] Could not determine max_level for {slide_name}, skipping.')
            continue

        images, coords, diagnosis = utils.load_slide_data(
            sqlite_path=sqlite_path,
            level=max_level,
            size=args.tilesize
        )

        if images is None or len(images) == 0:
            print(f'[WARNING] No images for slide {slide_name}, at level {max_level}, skipping.')
            continue
        if coords is None or len(coords) == 0:
            print(f'[WARNING] No coords for slide {slide_name}, at level {max_level}, skipping.')
            continue

        # Use diagnosis from sqlite as label
        if diagnosis is None or (isinstance(diagnosis, str) and diagnosis.strip() == ''):
            label = "unknown"
        else:
            label = str(diagnosis)

        # Check if output already exists
        output_file_path = os.path.join(encoder_out_path, f'{slide_name}_{label}.h5')
        if os.path.exists(output_file_path):
            print(f'[SKIP] Features already extracted for {slide_name} with label {label}.')
            continue

        print(f'No. of patches: {len(images)}')

        extract_Features(
            images=images,
            coords=coords,
            slide_name=slide_name,
            label=label,
            model=model,
            transform=transform,
            ndim=ndim,
            device=device,
            args=args
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
