from argparse import ArgumentParser

import numpy as np


from torch.utils.data import DataLoader, random_split



from src.datasets.TFPatchDataset import TFPatchDataset





def train(datasets, batch_size, n_epochs):
    params_train = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 6
    }
    train_gen = data.DataLoader(datasets['train'], **params_train)
    val_gen   = data.DataLoader(datasets['val'], **params_train)

def get_dataset(feature_dir, labels, patch_len, patch_hop):
    dataset = TFPatchDataset(feature_dir, label_dir, patch_len, patch_hop)
    return dataset

def split_dataset(dataset, val_split):
  ''' Generate training and validation torch.data.Dataset instances
      given a torch.data.Dataset and validation split proportion
  '''
  dataset_len = len(dataset)
  n_val = int(val_split * dataset_len)a
  n_train = dataset_len - n_val

  train_data, val_data = random_split(dataset, [n_train, n_val])
  return train_data, val_data


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


if __name__ == "__main__":
    args = parse_args()
    
    # dataset loader
    # split datar
    # model
    dataset = get_dataset(
                  args.feature_dir, 
                  args.label_dir, 
                  args.patch_len,
                  args.patch_hop
              )
    datasets['train'], datasets['val'] = split_dataset(dataset, args.val_split)

    train(
        datasets, 
        args.batch_size,
        args.n_epochs
    )
