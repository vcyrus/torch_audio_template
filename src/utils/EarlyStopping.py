import os

import torch

import numpy as np

'''
  https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
  A class to handle EarlyStopping, saving model checkpoints
'''

class EarlyStopping:
    """ Early stops if validation loss does not improve after patience """
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience 
        self.verbose = verbose
        self.delta = delta

        self.early_stop = False
        self.val_loss = None
        self.counter = 0
        self.val_loss_min = np.Inf

        if not os.path.exists('checkpoints/'):
            os.mkdir('checkpoints')


    def __call__(self, val_loss, model):
        if self.val_loss is None:
            self.val_loss = val_loss
            self.model_state = model.state_dict()
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.val_loss + self.delta:
            self.counter += 1
            print(f"Early Stopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.val_loss = val_loss
            self.counter = 0
            self.save_checkpoint(val_loss, model)
            self.model_state = model.state_dict()
            

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) . Saving model ...')
        torch.save(model.state_dict(), 'checkpoints/checkpoint.pt')
        self.val_loss_min = val_loss
