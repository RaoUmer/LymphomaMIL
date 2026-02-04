
import argparse
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
import torch

import h5py
#from CONCH.conch.open_clip_custom import create_model_from_pretrained
from conchv1_5.conchv1_5 import create_model_from_pretrained


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


class Conch(BaseEncoder):
    def __init__(self, name):
        super().__init__(name)

    def _build(self, img_size=448):

        #model, transform = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conchv1_5")
        model, transform = create_model_from_pretrained(checkpoint_path="hf_hub:MahmoodLab/conchv1_5", img_size=img_size)
        model.eval()

        return model, transform

    def forward(self, imgs):
        return self.model(imgs)