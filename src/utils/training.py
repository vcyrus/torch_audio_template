import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from src.models.SOLCNN import SOLCNN

from src.utils.EarlyStopping import EarlyStopping

def train(datasets, batch_size, n_epochs, lr, cuda_device, out_path, fs,
          win_length, hop_length, n_bins, n_classes, writer,
          dropout_rate, weight_decay, audio_len_s): 
    '''
      datasets   -> dict of {'train': data.DataLoader, 'val': data.DataLoader}
      batch_size -> int
      n_epochs   -> int, number of training epochs
      lr         -> float, learning rate
      use_cuda   -> bool, use gpu
      out_path   -> str, path to write the model to
      fs         -> int, audio sampling rate
      win_length -> int, feature extraction window length (ms)
      hop_length -> int, feature extraction hop length (ms)
      n_bins     -> int, feature extraction number of mel bands
      n_classes  -> int, number of classification classes
      writer     -> torch.utils.tensorboard.SummaryWriter
      dropout_rate -> float, dropout rate for model
      weight_decay -> float, loss regularization weight decay
      audio_len_s  -> int, length of the input audio in seconds
    '''
    params_train = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 6
    }
    generators = {
        'train': DataLoader(datasets['train'], **params_train),
        'val': DataLoader(datasets['val'], **params_train)
    }
  
    win_length = int((win_length / 1000) * fs)
    hop_length = int((hop_length / 1000) * fs)
    model = SOLCNN(
                fs, win_length, hop_length, n_bins, n_classes, 
                dropout_rate, audio_len_s
            )
    model.to(cuda_device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3)
    patience = 5
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(n_epochs):
        for phase in ['train', 'val']:
            loss = run_epoch(
                model, criterion, optimizer, 
                generators[phase], cuda_device, phase
            )
            log(epoch + 1, n_epochs, phase, loss, writer)
            if phase == 'val':
                # update the learning rate scheduler and save to history
                scheduler.step(loss)
                early_stopping(loss, model)

        if early_stopping.early_stop:
            print("Early Stopping")
            break
    
    print('------------------------------')
    print('Finished training')
    print('------------------------------')
    # Save the model
    if out_path:
        torch.save(model.state_dict(), out_path)

    return model, criterion

def run_epoch(model, criterion, optimizer, generator, cuda_device, phase=None):
    '''
       Passes through an train or val phase for a single epoch
       phase == 'train' then no gradient propagated
       generator: {data.DataLoader} either train or val generator
       cuda_device: {torch.device} 
       phase: 'train' or 'val'
    '''
    running_loss = 0.0 # running loss across batches
    n_batches = 0     # accumulate total batches to average the loss

    # initialise the model to not use dropout etc on eval
    model = model.train() if phase =='train' else model.eval()
    for i, batch in enumerate(generator, 0):
        
        X, y = batch["audio"], batch["label"]
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

def log(epoch, n_epochs, phase, loss, writer):
    print('%d / %d %s loss: %.5f' % (epoch , n_epochs, phase, loss))
    
    writer.add_scalar('Loss/{}'.format(phase), loss, epoch)
