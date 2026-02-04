import argparse
import os
import time
from abc import ABC, abstractmethod

import h5py
import torch
import torchvision.transforms.v2 as T

class BaseEncoder(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model, self.transform = self._build()

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def forward(self, imgs):
        pass

    def get_transform(self):
        return self.transform

class H0mini(BaseEncoder):
    def __init__(self, name):
        super().__init__(name)

    def _build(self):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.eval()
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        return model, transform

    def forward(self, imgs):
        output = self.model(imgs)
        # CLS token features (1, 768):
        cls_features = output[:, 0]
        # Patch token features (1, 256, 768):
        patch_token_features = output[:, self.model.num_prefix_tokens:]
        # Concatenate the CLS token features with the mean of the patch token
        # features (1, 1536):
        concatenated_features = torch.cat(
            [cls_features, patch_token_features.mean(1)], dim=-1
        )

        return concatenated_features

class HOptimus1(BaseEncoder):
    def __init__(self, name):
        super().__init__(name)

    def _build(self):
        import timm
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-1", pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
        model.eval()

        transform = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        return model, transform

    def forward(self, imgs):
        features = self.model(imgs)
        
        return features

def get_model(name: str):
    if name == "h0-mini":
        model = H0mini(name)
    elif name == "h-optimus-1":
        model = HOptimus1(name)
    else:
        raise ValueError("Invalid model name")

    return model, model.get_transform()


