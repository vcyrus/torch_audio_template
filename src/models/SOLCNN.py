import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import MelSpectrogram
from nnAudio import Spectrogram

import numpy as np

from src.models.Model import Model

from src.utils.calc_dims import calculate_out_dims

class SOLCNN(Model):
    def __init__(self, 
                fs, 
                win_length, 
                hop_length, 
                n_bins, 
                n_classes, 
                p_dropout, 
                audio_len_s, 
                n_channels_conv1=32, 
                n_channels_conv2=32,
                transform='mel_spec'):
        super(SOLCNN, self).__init__()

        if transform == 'mel_spec':
            self.transform = MelSpectrogram(sample_rate=fs,
                                          n_fft=2048,
                                          win_length=win_length,
                                          hop_length=hop_length,
                                          n_mels=n_bins)
        if transform == 'cqt':
            self.transform = Spectrogram.CQT(sr=fs,
                                            hop_length=hop_length,
                                            fmin=55,
                                            fmax=None,  
                                            n_bins=n_bins,
                                            bins_per_octave=12,
                                            norm=1,
                                            window='hann',
                                            center=True,
                                            pad_mode='reflect')
        
        self.bn1 = nn.BatchNorm2d(1)

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=n_channels_conv1, 
            kernel_size=5, stride=1
        )
        self.bn2 = nn.BatchNorm2d(n_channels_conv1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,2), stride=(4,2))

        self.conv2 = nn.Conv2d(
            in_channels=n_channels_conv1, 
            out_channels=n_channels_conv2, 
            kernel_size=5, stride=1
        )
        self.bn3 = nn.BatchNorm2d(n_channels_conv2)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,2), stride=(4,2))

        ''' 
          batchnorm-relu to be placed between each conv-bn-relu-pool block
          as pre-activation
        '''
        self.bn4 = nn.BatchNorm2d(n_channels_conv1)
        self.bn5 = nn.BatchNorm2d(n_channels_conv2)

        ''' 
          calculate the number of output dimensions after flattening 
          the last conv layer 
        '''
        n_frames = np.ceil((audio_len_s * fs + hop_length) / hop_length) 
        input_dims = (-1, 1, n_bins, n_frames)
        n_channels, h, w = calculate_out_dims(
                            input_dims, 
                            [self.conv1, self.pool1, self.conv2, self.pool2],
                            n_channels_conv2
                        )
        n_dims = n_channels*h*w

        self.dropout1 = nn.Dropout(p=p_dropout)
        self.fc1 = nn.Linear(in_features=n_dims, out_features=64)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(in_features=64, out_features=n_classes)

    def forward(self, batch):
        x = self.transform(batch)
        if len(x.size()) == 3:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(x))
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.dropout1(x.view(x.size(0), -1))
        x = F.relu(self.fc1(x))
        y = F.log_softmax(self.fc2(self.dropout2(x)), dim=1)
        return y
