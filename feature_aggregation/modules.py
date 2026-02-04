import os
import numpy as np
import torch
import torch.nn as nn
# Import aggregation methods
from aggregator import (
    GMA, TransMIL, TransMILBEL, MixtureMIL, DSMIL, CLAM_SB, CLAM_MB, VarAttention,
    GTP, PatchGCN_Surv, DeepGraphConv_Surv, MIL_Sum_FC_surv, MIL_Attention_FC_surv,
    MIL_Cluster_FC_surv, PureTransformer, ViT, Transformer
)

def get_aggregator(method, ndim, n_classes, **kwargs):
    """
    Only pass the arguments each model actually supports.
    """
    m = method.lower()

    # ---------------- basic MILs ----------------
    if m == 'ab-mil':
        return GMA(ndim=ndim, n_classes=n_classes)
    elif m == 'ab-mil_fc_small':
        return MIL_Attention_FC_surv(ndim=ndim, n_classes=n_classes, size_arg='small')
    elif m == 'ab-mil_fc_big':
        return MIL_Attention_FC_surv(ndim=ndim, n_classes=n_classes, size_arg='big')
    elif m == 'varmil':
        return VarAttention(ndim=ndim)

    # ---------------- CLAM ----------------
    elif m == 'clam_sb':
        return CLAM_SB(ndim=ndim)
    elif m == 'clam_mb':
        return CLAM_MB(ndim=ndim)

    # ---------------- ViT / Transformers ----------------
    elif m == 'vit_mil':
        return PureTransformer(ndim=ndim, n_classes=n_classes)
    elif m == 'vit':
        return ViT(ndim=ndim, n_classes=n_classes)
    elif m == 'transformer':
        return Transformer(ndim=ndim, n_classes=n_classes)

    # ---------------- TransMIL family ----------------
    elif m == 'transmil':
        return TransMIL(ndim=ndim, n_classes=n_classes)
    elif m == 'transmilbel':
        # Only TransMILBEL needs margin & lamda; note it expects n_features=ndim.
        margin = kwargs.get('margin', 0.1)
        lamda  = kwargs.get('lamda', 0.996)
        #print('module: ndim:', ndim)
        return TransMILBEL(n_classes=n_classes, n_features=ndim, margin=margin, lamda=lamda)

    # ---------------- GNNs / others ----------------
    elif m == 'ds-mil':
        return DSMIL(ndim=ndim, n_classes=2)
    elif m == 'gtp':
        return GTP(ndim=ndim)
    elif m == 'patchgcn':
        return PatchGCN_Surv(ndim=ndim, n_classes=2)
    elif m == 'deepgraphconv':
        return DeepGraphConv_Surv(ndim=ndim, n_classes=2)
    
    # ---------------- Simple linear classifier ----------------
    elif m == 'linear':
        # Assumes input is a single slide / bag embedding of size ndim
        # and outputs logits over n_classes.
        return nn.Linear(ndim, n_classes)

    else:
        raise Exception(f'Method {method} not defined')
