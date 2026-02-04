import os
import timm
import torch
import torch.nn as nn

class kaiko_vitl14(nn.Module):
    def __init__(self):
        super(kaiko_vitl14, self).__init__()
        self.model = timm.create_model(model_name="hf-hub:1aurent/vit_large_patch14_reg4_224.kaiko_ai_towards_large_pathology_fms", 
                                        dynamic_img_size=True,
                                        pretrained=True,)
    def forward(self, x):
        embedding = self.model(x)
        return embedding