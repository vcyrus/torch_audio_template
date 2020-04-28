import os

import numpy as np

import torch
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam  

from src.models.SOLCNN import SOLCNN

from src.utils.training import train, get_datasets
from src.utils.parse_args import parse_args
from src.utils.test import evaluate

if __name__ == "__main__":
    args = parse_args()
    
    datasets, label_to_int = get_datasets(
                  args.audio_path, 
                  args.fs,
                  args.audio_len,
                  args.val_split,
                  args.test_split
              )
    n_classes = len(label_to_int)

    writer = SummaryWriter()

    cuda_device = torch.device('cuda' if args.use_cuda and \
                                      torch.cuda.is_available() \
                                      else 'cpu')
    print('Device: {}'.format(cuda_device))
    win_length = int((args.win_length / 1000) * args.fs)
    hop_length = int((args.hop_length / 1000) * args.fs)
    model = SOLCNN(args.fs, 
                  win_length, 
                  hop_length, 
                  args.n_bins, 
                  n_classes, 
                  args.dropout_rate, 
                  args.audio_len,
                  transform=args.transform)  

    optimizer = Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    model, criterion = train(
        datasets, 
        args.batch_size,
        args.n_epochs,
        args.lr,
        cuda_device,
        args.fs,
        model,
        optimizer,
        writer,
        args.weight_decay, 
        args.audio_len,
        criterion,
        args.patience,
    )
    writer.close()

    # Save the model
    if args.out_path:
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
        path = os.path.join(args.out_path, 
                          "model_{0}".format(args.transform))
        torch.save(model.state_dict(), path)

    confusions = evaluate(model, 
                          cuda_device, 
                          datasets['test'], 
                          label_to_int, 
                          criterion)
