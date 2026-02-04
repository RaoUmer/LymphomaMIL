import torch
from transformers import AutoModel


class Titan(torch.nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True,
                 precision: torch.dtype = torch.float32):
        super().__init__()
        assert pretrained, "TitanSlideEncoder supports pretrained=True only."
        self.model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
        self.embedding_dim = 768
        self.precision = precision  # use fp32 to avoid half-precision issues

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def forward(self, batch, psize: int = 512, device: str = 'cuda'):
        """
        Expects:
          batch['features']:  (N, D) tensor
          batch['coords']:    (N, 2) tensor (x,y at level-0)

        psize: patch_size_lv0 = distance between adjacent patches at level 0.
               E.g., 512 (20x) or 1024 (40x).
        """
        feats = batch['features'].to(device, dtype=self.precision)
        coords = batch['coords'].to(device)

        patch_size_lv0 = int(psize)  # TITAN expects a scalar here

        slide_emb = self.model.encode_slide_from_patch_features(
            feats, coords, patch_size_lv0
        )

        return slide_emb
