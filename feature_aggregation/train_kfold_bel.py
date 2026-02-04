import datasets
import modules
import os
import argparse
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pdb
import random

from sklearn.metrics import f1_score

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
    'titan'
    ], help='which encoder to use')
parser.add_argument('--method', type=str, default='', choices=[
    'AB-MIL',
    'AB-MIL_FC_small',
    'AB-MIL_FC_big',
    'CLAM_SB',
    'CLAM_MB',
    'transMIL',
    'transMILBEL',
    'Transformer',
    'ViT',
    'DS-MIL',
    'VarMIL',
    'GTP',
    'PatchGCN',
    'DeepGraphConv',
    'ViT_MIL',
    'DTMIL',
    'LongNet_ViT',
    'Linear'
    ], help='which aggregation method to use')
parser.add_argument('--kfold', default=0, type=int, choices=list(range(0,5)), help='which 0 t0 5?')
parser.add_argument('--num_classes', default=2, type=int, help='number of classes')

# OPTIMIZATION PARAMS
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument('--lr_end', type=float, default=1e-6)
parser.add_argument("--warmup_epochs", default=10, type=int)
parser.add_argument('--weight_decay', type=float, default=0.04)
parser.add_argument('--weight_decay_end', type=float, default=0.4)
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--random_seed', default=0, type=int)

# Bag Embedding loss hyperparams
parser.add_argument('--be_weight', type=float, default=1.0,
                    help='weight for BagEmbeddingLoss; set 0 to disable')
parser.add_argument('--be_margin', type=float, default=0.175,
                    help='margin for cosine embedding in BagEmbeddingLoss')
parser.add_argument('--be_momentum', type=float, default=0.996,
                    help='EMA momentum (lambda) for BE dictionary updates')

def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    global args
    args = parser.parse_args()
    if args.random_seed:
        set_random_seed(args.random_seed)

    args.save_dir = os.path.join(args.output, args.method)
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f'convergence_kfold{args.kfold}.csv')

    # Label dictionary (example for Munich)
    label_dict = {'CLL': 0, 'FL': 1, 'MCL': 2, 'MAL': 2, 'NOS': 3, 'neg': 4}

    # Datasets
    train_dset, val_dset, test_dset = datasets.get_datasets_kfold(
        kfold=args.kfold, data=args.data, encoder=args.encoder, label_dict=label_dict
    )
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.workers)

    # Dim of features based on encoder
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
    elif args.encoder == 'kaiko_vitl14':
        ndim = 1024
    elif args.encoder == 'titan':
        ndim = 768
    elif args.encoder == 'dinosmall':
        ndim = 384
    elif args.encoder == 'dinobase':
        ndim = 768
    else:
        raise Exception('Wrong encoder name')
    
    #print('ndim:', ndim)
    # Model
    model = modules.get_aggregator(method=args.method,
                                    ndim=ndim, 
                                    n_classes=args.num_classes, 
                                    margin=args.be_margin, 
                                    lamda=args.be_momentum)
    model.cuda()

    # Losses
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer & schedulers
    params_groups = get_params_groups(model)
    optimizer = optim.AdamW(params_groups)
    lr_schedule = cosine_scheduler(args.lr, args.lr_end, args.nepochs, len(train_loader),
                                   warmup_epochs=args.warmup_epochs)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end,
                                   args.nepochs, len(train_loader))
    cudnn.benchmark = True

    with open(log_path, 'w') as fconv:
        fconv.write('epoch,metric,value\n')

    best_f1 = 0.0

    # Train
    for epoch in range(args.nepochs+1):
        if epoch > 0:
            loss = train(epoch, train_loader, model, criterion,
                         optimizer, lr_schedule, wd_schedule, be_weight=args.be_weight)
            print(f'Training\tEpoch: [{epoch}/{args.nepochs}]\tLoss: {loss}')
            with open(log_path, 'a') as fconv:
                fconv.write(f"{epoch},loss,{loss}\n")

        # Validate
        probs = test(epoch, val_loader, model, args)
        y_true = val_loader.dataset.df['target'].map(label_dict).values
        y_pred = np.argmax(probs, axis=1)
        f1 = f1_score(y_true, y_pred, average='macro')

        if f1 > best_f1:
            print(f"New best F1 model found at epoch {epoch} with F1: {f1:.4f}")
            best_f1 = f1
            obj = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'f1': f1,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.save_dir, f'checkpoint_best_f1_kfold{args.kfold}.pth'))

        with open(log_path, 'a') as fconv:
            fconv.write(f"{epoch},f1,{f1}\n")

        obj = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'f1': f1,
            'optimizer': optimizer.state_dict()
        }
        torch.save(obj, os.path.join(args.save_dir, f'checkpoint_latest_kfold{args.kfold}.pth'))

def test(run, loader, model, args):
    model.eval()
    probs = torch.FloatTensor(len(loader), args.num_classes).cuda()
    with torch.no_grad():
        for i, (input, _) in enumerate(loader):
            print(f'Inference\tEpoch: [{run}/{args.nepochs}]\tBatch: [{i+1}/{len(loader)}]')
            feat = input.float().cuda() # .unsqueeze(0) for titan
            #print('tt feat:', feat.shape)
            logits, _ = model(feat, label=None)
            Y_prob = torch.softmax(logits, dim=-1)
            if Y_prob.dim() > 1 and Y_prob.size(1) == 1:
                Y_prob = Y_prob.squeeze(1)
            probs[i] = Y_prob.detach()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer,
          lr_schedule, wd_schedule, be_weight: float = 1.0):
    model.train()
    running_loss = 0.0
    for i, (input, target) in enumerate(loader):
        it = len(loader) * (run-1) + i
        for j, pg in enumerate(optimizer.param_groups):
            pg["lr"] = lr_schedule[it]
            if j == 0:
                pg["weight_decay"] = wd_schedule[it]

        feat = input.float().cuda() # .unsqueeze(0) for titan
        #print('tr feat:', feat.shape)
        target = target.long().cuda()

        logits, CELoss = model(feat, label=target)
        loss = criterion(logits, target) + be_weight * CELoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(f'Training\tEpoch: [{run}/{args.nepochs}]\tBatch: [{i+1}/{len(loader)}]\t'
              f'total loss: {loss.item():.4f}')
    return running_loss / len(loader)

def get_params_groups(model):
    regularized, not_regularized = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

if __name__ == '__main__':
    main()
