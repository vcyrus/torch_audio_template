from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.optim import Adam  

from src.datasets.TFPatchDataset import TFPatchDataset

from src.models.ConvNet import ConvNet

def train_epoch(model, criterion, optimizer, generator, cuda_device, epoch, phase=None):
    running_loss = 0.0
    model.train()
    n_batches = 0
    for i, batch in enumerate(generator, 0):
        
        X, y = batch["features"], batch["labels"]
        X = X.to(device=cuda_device)
        y = y.to(device=cuda_device)

        _y = model(X)
        loss = criterion(_y, y)
        
        if phase == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item() 
        
        n_batches += 1
    return running_loss / n_batches

def train(datasets, batch_size, n_epochs, lr, use_cuda, out_path):
    params_train = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 6
    }
    generators = {
        'train': DataLoader(datasets['train'], **params_train),
        'val': DataLoader(datasets['val'], **params_train)
    }
  
    cuda_device = torch.device('cuda' if use_cuda and torch.cuda.is_available() \
                                     else 'cpu')
    model = ConvNet()
    model.to(cuda_device)

    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr)

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            loss = train_epoch(
                model, criterion, optimizer, 
                generators[phase], cuda_device, epoch, phase
            )
            print('%d / %d %s loss: %.5f' % (epoch + 1, n_epochs, phase, loss))

        print("Epoch %d complete" % epoch)

    print('Finished training')

    # Save the model
    if out_path:
        torch.save(model.state_dict(), out_path)
            

def get_datasets(feature_dir, patch_len, patch_hop, val_split):
    # get file name bases from csv
    dataset = TFPatchDataset(feature_dir, patch_len, patch_hop)

    dataset_len = len(dataset)
    n_val = int(val_split * dataset_len)
    n_train = dataset_len - n_val
  
    train_data, val_data = random_split(dataset, [n_train, n_val])

    datasets = {'train': train_data, 'val': val_data}
    return datasets

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "feature_dir", type=str, help="path to features .npy files"
    ),
    parser.add_argument(
        "--patch_len", 
        default=128, 
        type=int, 
        help="input patch length"
    ),
    parser.add_argument(
        "--patch_hop", 
        default=128, 
        type=int, 
        help="input patch overlap"
    ),
    parser.add_argument(
        "--batch_size", 
        default=1, 
        type=int, 
        help="training batch size"
    ),
    parser.add_argument(
        "--val_split",
        default=0.1,
        type=float,
        help="Percentage of training data used for validation"
    ),
    parser.add_argument(
        "--n_epochs",
        default=10,
        type=int,
        help="Number of training epochs"
    ),
    parser.add_argument(
        "-lr",
        default=0.01, 
        type=float,
        help="Optimizer learning rate"
    ),
    parser.add_argument(
        "--use_cuda",
        default=False,
        type=bool,
        help="Use current CUDA GPU device if True, CPU if unspecified"
    ),
    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path to save the trained model to"
    )
    return parser.parse_args()  

if __name__ == "__main__":
    args = parse_args()
    
    datasets = get_datasets(
                  args.feature_dir, 
                  args.patch_len,
                  args.patch_hop,
                  args.val_split,
              )

    train(
        datasets, 
        args.batch_size,
        args.n_epochs,
        args.lr,
        args.use_cuda,
        args.out_path
    )
