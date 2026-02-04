import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, balanced_accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

# I/O PARAMS
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--data', type=str, default='', help='which data to use')
parser.add_argument('--encoder', type=str, default='', choices=[
    'resnet_50',
    'ctranspath',
    'phikon',
    'uni',
    'uni2',
    'virchow',
    'virchow2',
    'gigapath',
    'dinosmall',
    'dinobase',
    'kaiko_vitl14',
    'titan',
    'h0_mini',
    'hoptimus1'
], help='which encoder to use')
parser.add_argument('--method', type=str, default='', choices=[
    'AB-MIL',
    'AB-MIL_FC_small',
    'AB-MIL_FC_big',
    'CLAM_SB',
    'CLAM_MB',
    'transMIL',
    'Transformer',
    'DS-MIL',
    'VarMIL',
    'GTP',
    'PatchGCN',
    'DeepGraphConv',
    'ViT_MIL',
    'DTMIL'
    'LongNet_ViT'
], help='which aggregation method to use')
parser.add_argument('--kfold', default=0, type=int, choices=list(range(0,5)), help='which fold (0 to 5)')
parser.add_argument('--num_classes', default=5, type=int, help='number of classes')
parser.add_argument('--checkpoint', type=str, default='', help='path to the model checkpoint')
parser.add_argument('--log_csv', type=str, default='inference_results.csv', help='CSV file to save the results')

# OPTIMIZATION PARAMS (not used in this case, only for loading)
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 10)')

def main():
    # Get user input
    global args
    args = parser.parse_args()

    args.save_dir = os.path.join(args.output, args.method)
    os.makedirs(args.save_dir, exist_ok=True)  # ensure directory exists

    # Load datasets (train_dset is not used for inference)
    label_dict = {'CLL': 0, 'FL': 1, 'MCL': 2, 'MAL': 2, 'NOS': 3, 'neg': 4} # munich
    #label_dict = {'CLL': 0, 'FL': 1, 'MCL': 2, 'DLBCL': 3, 'Lts': 4} # regensburg
    train_dset, val_dset, test_dset = datasets.get_datasets_kfold(kfold=args.kfold, data=args.data, label_dict=label_dict)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=args.workers)

    # Determine model feature dimensions based on the encoder type
    if args.encoder == 'resnet_50':
        ndim = 1024
    elif args.encoder == 'ctranspath':
        ndim = 768
    elif args.encoder == 'phikon':
        ndim = 768
    elif args.encoder == 'uni':
        ndim = 1024
    elif args.encoder == 'uni2':
        ndim = 1536
    elif args.encoder == 'virchow':
        ndim = 2560
    elif args.encoder == 'virchow2':
        ndim = 2560
    elif args.encoder == 'gigapath':
        ndim = 1536
    elif args.encoder == 'h0_mini':
        ndim = 1536
    elif args.encoder == 'hoptimus1':
        ndim = 1536
    elif args.encoder == 'kaiko_vitl14':
        ndim = 1024
    elif args.encoder == 'titan':
        ndim = 768
    elif args.encoder.startswith('dinosmall'):
        ndim = 384
    elif args.encoder.startswith('dinobase'):
        ndim = 768
    else:
        raise Exception('Wrong encoder name')

    # Get model
    model = modules.get_aggregator(method=args.method, ndim=ndim, n_classes=args.num_classes)
    model.cuda()

    # Load pretrained model checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Checkpoint loaded, AUC at checkpoint: {checkpoint.get('auc', 'N/A')}")

    # Perform inference on test data and log results
    print("Starting inference...")
    results_df = test(0, test_loader, model, args)

    # Get the true labels and probabilities
    # Adjust label_dict to remove duplicate mappings (since MCL and MAL are treated as the same class)
    label_dict_renamed = {'CLL': 0, 'FL': 1, 'MCL/MAL': 2, 'NOS': 3, 'neg': 4} # munich
    #label_dict = {'CLL': 0, 'FL': 1, 'MCL': 2, 'DLBCL': 3, 'Lts': 4} # regensburg
    #label_dict_renamed = label_dict # regensburg 
    reverse_label_dict = {v: k for k, v in label_dict_renamed.items()}  # Reverse mapping for easy lookup

    # Map true labels, merging 'MCL' and 'MAL' into the same class
    true_labels = results_df['true_label'].replace({'MCL': 'MCL/MAL', 'MAL': 'MCL/MAL'}).map(label_dict_renamed).values # munich
    #true_labels = results_df['true_label'].map(label_dict_renamed).values # regensburg
    probs = np.stack(results_df['probabilities'].values)
    
    # AUC Score for multiclass
    auc = roc_auc_score(true_labels, probs, multi_class='ovr', average='macro')
    print(f'Inference AUC: {auc}')
    
    # Predicted labels (argmax to get the predicted class)
    predicted_labels = np.argmax(probs, axis=1)

    # F1-macro score
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    print(f'Inference F1-macro: {f1_macro}')

    # Precision score
    precision = precision_score(true_labels, predicted_labels, average='macro')
    print(f'Inference Precision: {precision}')

    # Recall score
    recall = recall_score(true_labels, predicted_labels, average='macro')
    print(f'Inference Recall: {recall}')

    # Accuracy
    bacc = balanced_accuracy_score(true_labels, predicted_labels)
    print(f'Inference Balanced Accuracy: {bacc}')

    # Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Save the results (including per-image probabilities, true labels, predicted labels)
    results_df['predicted_label'] = predicted_labels
    results_df['auc'] = auc
    results_df['f1_macro'] = f1_macro
    results_df['bacc'] = bacc
    results_df['precision'] = precision
    results_df['recall'] = recall

    # Save the results to CSV
    results_df.to_csv(os.path.join(args.save_dir, '{}_inference_results_kf{}.csv'.format(args.encoder, args.kfold)), index=False)

    # Save confusion matrix to CSV as well
    #conf_matrix_df = pd.DataFrame(conf_matrix, index=list(label_dict.keys()), columns=list(label_dict.keys()))
    conf_matrix_df = pd.DataFrame(conf_matrix, index=list(reverse_label_dict.values()), columns=list(reverse_label_dict.values()))
    conf_matrix_df.to_csv(os.path.join(args.save_dir, '{}_confusion_matrix_kf{}.csv'.format(args.encoder, args.kfold)), index=True)
    
    print(f'Results saved to {args.log_csv} and confusion matrix saved to confusion_matrix.csv')


def test(run, loader, model, args):
    # Set model in test mode
    model.eval()

    # Initialize a list to store probabilities, true labels, and other details for each image
    results = []

    # Loop through batches
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            print('Inference\tBatch: [{}/{}]'.format(i+1, len(loader)))
            
            # Copy batch to GPU
            feat = input.float().cuda() # .unsqueeze(0) for titan + transmil, and squeeze(0) for abmil
            #print('feat:', feat.shape)
            results_dict = model(feat)
            #print('tt::output:', output.shape, output.min(), output.max())
            logits, Y_prob, Y_hat = (results_dict[key] for key in ['logits', 'Y_prob', 'Y_hat'])

            # Store the probabilities and true label
            results.append({
                'patient_id': loader.dataset.df.iloc[i]['slide'],  # Assuming the 'slide' column holds the image/slide ID
                'true_label': loader.dataset.df.iloc[i]['target'],  # True label
                'probabilities': Y_prob.detach().cpu().numpy().squeeze()  # Predicted probabilities
            })
            torch.cuda.empty_cache()

    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == '__main__':
    main()
