import os
import resnet_custom
import vision_transformer as vits
import uni
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import kaiko
import titan
import conch

class tres50(nn.Module):
    def __init__(self):
        super(tres50, self).__init__()
        resnet = resnet_custom.resnet50_baseline(True)
        self.features = nn.Sequential(*list(resnet.children())[0:-1])
    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return x       
    
def load_pretrained_weights_dino(model, pretrained_weights, checkpoint_key):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))

def get_encoder(encoder):
    
    if encoder == 'tres50_imagenet':
        model = tres50()
        ndim = 1024
    elif encoder == 'res50_imagenet':
        model = models.__dict__['resnet50'](weights='DEFAULT')
        model.fc = nn.Identity()
        ndim = 2048
    elif encoder == 'ctranspath':
        from ctran import ctranspath
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'path/to/checkpoint.pth')
        model.load_state_dict(td['model'], strict=True)
        ndim = 768
    elif encoder == 'phikon':
        import phikon
        model = phikon.get_model()
        ndim = 768
    elif encoder == 'uni':
        model = uni.get_model(enc_name='uni')
        ndim = 1024
    elif encoder == 'uni2':
        model = uni.get_model(enc_name='uni2-h')
        ndim = 1536
    elif encoder == 'virchow':
        import virchow
        model = virchow.virchow()
        ndim = 2560
    elif encoder == 'virchow2':
        import virchow
        model = virchow.virchow2()
        ndim = 2560
    elif encoder == 'h0-mini':
        import optimus
        model, transform = optimus.get_model(encoder)
        ndim = 1536
    elif encoder == 'h-optimus-1':
        import optimus
        model, transform = optimus.get_model(encoder)
        ndim = 1536
    elif encoder == 'dinosmall':
        path = 'path/to/checkpoint.pth'
        model = vits.vit_small(num_classes=0)
        load_pretrained_weights_dino(model, path, 'teacher')
        ndim = 384
    elif encoder == 'dinobase':
        path = 'path/to/checkpoint.pth'
        model = vits.vit_base(num_classes=0)
        load_pretrained_weights_dino(model, path, 'teacher')
        ndim = 768
    elif encoder == 'gigapath':
        import timm
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        ndim = 1536
    elif encoder == 'kaiko_vitl14':
        model = kaiko.kaiko_vitl14()
        ndim = 1024
    elif encoder == 'conchv1_5':
        model = conch.Conch(name='conchv1_5')
        ndim = 768
    elif encoder == 'titan':
        model = titan.Titan(pretrained=True, freeze=True, precision=torch.float32)
        ndim = 768
    else:
        raise Exception('Wrong encoder name')
    
    if encoder == 'virchow':
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        transform = create_transform(**resolve_data_config(model.virchow.pretrained_cfg, model=model.virchow))
    elif encoder == 'virchow2':
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        transform = create_transform(**resolve_data_config(model.virchow2.pretrained_cfg, model=model.virchow2))
    elif encoder == 'h-optimus-0':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
    elif encoder == 'kaiko_vitl14':
         transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    elif encoder == 'conch':
        transform = model.get_transform()
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {encoder} - {total_params} parameters')
    return model, transform, ndim
