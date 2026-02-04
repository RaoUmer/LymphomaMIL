import os
import numpy as np
import pandas as pd
import torch
import h5py

def _normalize_label_dict(label_dict):
    """
    Make label_dict tolerant to int/str mismatches by adding mirrored keys.
    E.g., if key 4 exists, we add '4' -> same value; if '4' exists, add 4 -> same value.
    """
    if label_dict is None:
        return None
    norm = dict(label_dict)
    for k, v in list(label_dict.items()):
        if isinstance(k, int):
            norm[str(k)] = v
        elif isinstance(k, str) and k.isdigit():
            norm[int(k)] = v
    return norm

def get_datasets_kfold(kfold=0, data='', label_dict=None):
    # Load slide data
    df = pd.read_csv(data)

    # Clean/rename
    df = df.rename(columns={
        'patient_id': 'slide',
        'label': 'target',
        f'kfold{kfold}': 'kfoldsplit',
        'tensor_paths': 'tensor_path'
    })[['slide', 'target', 'kfoldsplit', 'tensor_path']]

    # If label_dict not provided, build from unique targets in CSV
    if label_dict is None:
        uniques = pd.unique(df['target'])
        # preserve numeric if possible
        try:
            as_int = pd.Series(uniques).astype(int)
            # numeric labels present -> identity mapping
            label_dict = {int(x): int(x) for x in as_int}
        except Exception:
            # non-numeric labels -> alphabetical mapping
            uniques_sorted = sorted(map(str, uniques))
            label_dict = {lab: i for i, lab in enumerate(uniques_sorted)}

    label_dict = _normalize_label_dict(label_dict)

    # Split
    df_train = df[df.kfoldsplit == 'train'].reset_index(drop=True).drop(columns=['kfoldsplit'])
    df_val   = df[df.kfoldsplit == 'val'  ].reset_index(drop=True).drop(columns=['kfoldsplit'])
    df_test  = df[df.kfoldsplit == 'test' ].reset_index(drop=True).drop(columns=['kfoldsplit'])

    # Some quick visibility (safe to keep; comment out if noisy)
    print(f"[datasets] label_dict keys: {list(label_dict.keys())}")
    print(f"[datasets] train targets unique: {pd.unique(df_train['target'])}")
    print(f"[datasets] val targets unique:   {pd.unique(df_val['target'])}")
    print(f"[datasets] test targets unique:  {pd.unique(df_test['target'])}")

    dset_train = slide_dataset_classification(df_train, label_dict)
    dset_val   = slide_dataset_classification(df_val, label_dict)
    dset_test  = slide_dataset_classification(df_test, label_dict)

    return dset_train, dset_val, dset_test

class slide_dataset_classification(object):
    """
    Slide-level dataset which returns, for each slide, the feature matrix (h) and the target.
    Expects each HDF5 to have a dataset at key 'features' with shape [N_patches, ndim].
    """
    def __init__(self, df, label_dict):
        self.df = df
        self.label_dict = label_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Load feature matrix
        with h5py.File(row.tensor_path, 'r') as f:
            if 'features' not in f:
                raise KeyError(f"'features' not found in {row.tensor_path}. Keys: {list(f.keys())}")
            feat = f['features'][:].astype(np.float32)  # ensure float32

        # Map label robustly (handle int/str)
        raw = row.target
        if raw in self.label_dict:
            target = self.label_dict[raw]
        elif isinstance(raw, str) and raw.isdigit() and (int(raw) in self.label_dict):
            target = self.label_dict[int(raw)]
        elif isinstance(raw, (int, np.integer)) and (str(raw) in self.label_dict):
            target = self.label_dict[str(raw)]
        else:
            raise ValueError(
                f"Label '{raw}' not found in label_dict keys {list(self.label_dict.keys())}. "
                "Check for type mismatches or unexpected class names in your CSV."
            )

        return feat, target
