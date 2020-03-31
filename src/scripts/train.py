from argparse import ArgumentParser

import numpy as np


from torch.utils.data import DataLoader, random_split


from src.datasets.TFPatchDataset import TFPatchDataset

from src.models import ConvNet

def train_epoch(model, criterion, optimizer, train_gen, cuda_device):
    running_loss = 0.0
    for i, batch in enumerate(train_gen, 0):
        optimiser.zero_grad()

        features, y = batch  

        if cuda_device is not None:
            features = features.to(device=cuda_device)
            y = labels.to(device=cuda_device)
        
        _y = model(features)
        loss = criterion(_y, y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss_item 
        if i % 100 == 99: # print loss after 100 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
    return none

def train(datasets, batch_size, n_epochs, lr, gpu, out_path):
    params_train = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 6
    }
    train_gen = data.DataLoader(datasets['train'], **params_train)
    val_gen   = data.DataLoader(datasets['val'], **params_train)
  
    cuda_device = cuda.device('cuda' if gpu else 'cpu')

    model = ConvNet()
    model.to(cuda_device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(n_epochs):
        train_epoch(model, criterion, optimizer, train_gen, cuda_device)

        # validation
        print("Epoch %d/%d - loss: %.3f, val_loss: %.3f" % 
              (epoch + 1, n_epochs, train_loss, val_loss)


    print('Finished training')

    # Save the model
    if out_path:
        torch.save(model.state_dict(), out_path)
            

def get_datasets(feature_dir, labels, patch_len, patch_hop, val_split):
    dataset = TFPatchDataset(feature_dir, label_dir, patch_len, patch_hop)

    dataset_len = len(dataset)
    n_val = int(val_split * dataset_len)
    n_train = dataset_len - n_val
  
    train_data, val_data = random_split(dataset, [n_train, n_val])

    datasets = {'train': train_data, 'val': val_data}
    return datasets

def parse_args():
    parser = ArgumentParser(
        parser.add_argument("feature_dir", type=str),
        parser.add_argument("label_dir", type=str),
        parser.add_argument(
            "--patch_len", 
            default=100, 
            type=int, 
            help="input patch length"
        ),
        parser.add_argument(
            "--patch_hop", 
            default=100, 
            type=int, 
            help="input patch overlap"
        )
        parser.add_argument(
            "--batch_size", 
            default=1, 
            type=int, 
            help="training batch size"
        )
        parser.add_argument(
            "--val_split",
            default=0.1
            type=float
            help="Percentage of training data used for validation"
        )
        parser.add_argument(
            "--n_epochs",
            default=10
            type=int
            help="Number of training epochs"
        )
        parser.add_argument(
            "-lr",
            default=0.01, 
            type=float,
            help="Optimizer learning rate"
        )
        parser.add_argument(
            "-gpu",
            default=False,
            type=bool,
            help="Use current CUDA GPU device if True, CPU if unspecified"
        )
        parser.add_argument(
            "--out_path",
            default=None,
            type=str,
            help="Path to save the trained model to"
        )

if __name__ == "__main__":
    args = parse_args()
    
    # dataset loader
    # split datar
    # model
    datasets = get_datasets(
                  args.feature_dir, 
                  args.label_dir, 
                  args.patch_len,
                  args.patch_hop,
                  args.val_split
              )

    train(
        datasets, 
        args.batch_size,
        args.n_epochs,
        args.lr,
        args.gpu,
        args.out_path
    )
