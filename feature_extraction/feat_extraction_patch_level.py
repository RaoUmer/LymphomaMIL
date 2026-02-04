import glob
import os
import pandas as pd
import gc
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import encoders
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='', help='path to data')
parser.add_argument('--csv_path', type=str, default='', help='path to csv data')
parser.add_argument('--out_path', type=str, default='', help='path to output data')
parser.add_argument('--encoder', type=str, default='', choices=[
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
    'conch'
], help='choice of encoder')
parser.add_argument('--tilesize', type=int, default=224, help='tile size')
parser.add_argument('--workers', type=int, default=10, help='workers')


def read_patches(path):
    patch_data_list = []
    # Collect .png and .jpg file paths
    patch_paths = glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.jpg'))
    
    for patch_path in patch_paths:
        patch_img = Image.open(patch_path)
        
        # Convert CMYK or RGBA to RGB
        if patch_img.mode == 'CMYK':
            patch_img = patch_img.convert('RGB')
        if patch_img.mode == 'RGBA':
            patch_img = patch_img.convert('RGB')
        
        # Convert image to a NumPy array
        patch_img = np.array(patch_img)
        patch_data_list.append(patch_img)
    
    return patch_data_list


def extract_Features(patch_data_list, slide_name, label, model, transform, ndim, device, args):
    #create output folder
    if not os.path.exists(os.path.join(args.out_path, args.encoder)):
        os.mkdir(os.path.join(args.out_path, args.encoder))
    
    #empty feature variables
    feats= np.empty([0, ndim])
    #iterate over files of one patient
    with torch.no_grad():
        for patch in patch_data_list:
            img = patch
            # resizing
            img = cv2.resize(img, dsize=(args.tilesize, args.tilesize), interpolation=cv2.INTER_CUBIC)
            # Convert the resized image (NumPy array) to a PIL Image
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #prepare patch                                
            img_t = transform(img)
            batch_t = torch.unsqueeze(img_t, 0)
            batch_t = batch_t.to(device)
            #extract features with model
            features = model(batch_t)		
            features = features.cpu().detach().numpy()
            feats = np.append(feats,features,0)

    # Write data to HDF5
    print('features size: ', feats.shape)  
    file= h5py.File(args.out_path+'/'+args.encoder+'/'+slide_name+'_'+label+'.h5', 'w')
    file.create_dataset("features", data=feats)
    file.close()                                             

def main():
    '''
    feature extraction
    '''
    global args
    args = parser.parse_args()
    
    # Set up encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, ndim = encoders.get_encoder(args.encoder)
    model.to(device)
    
    # Set up data
    metadata = pd.read_csv(args.csv_path)
    
    slide_names = metadata.loc[:,'slide_id'].unique()
    slide_labels = metadata.loc[:,'label'].unique()

    print('unique ids:', len(slide_names))
    print('unique labels:', len(slide_labels))
    
    # Ensure the encoder-specific output directory exists
    encoder_out_path = os.path.join(args.out_path, args.encoder)
    if not os.path.exists(encoder_out_path):
        os.makedirs(encoder_out_path)

    for label in slide_labels:
        for slide_name in slide_names:
            # Construct the expected output file path
            output_file_path = os.path.join(encoder_out_path, f'{slide_name}_{label}.h5')
            if os.path.exists(output_file_path):
                print(f'Features already extracted for {slide_name} with label {label}, skipping.')
                continue  # Skip this file as features are already extracted
            patches_path = os.path.join(args.data_path, label, slide_name)
            print('patches path:', patches_path)
            if os.path.exists(patches_path):
                patch_data_list = read_patches(patches_path)
                print('No. of patches:', len(patch_data_list))
                extract_Features(patch_data_list, slide_name, label, model, transform, ndim, device, args)

if __name__ == '__main__':
    main()