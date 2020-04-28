'''
   SOLDataset.py: src.datasets.SOLDataset

   Loads SOL instrument audio clips with a specified length and labels.
   Assumes directory structure SOL_root/<instrument>/ where each
   /<instrument>/ contains .wav files

   Cyrus Vahidi
'''

import os

import torchaudio

from src.datasets.Dataset import Dataset

class SOLDataset(Dataset):
    ''' path:         absolute path to SOL instrument folders. should contain 
                      instrument name folders that contain .wav files
        label_to_int: label to categorical int map
        fs:           target sample rate
        suffix:       audio file suffix
        audio_len:    length of audio (seconds) to be generated
    '''

    def __init__(self, 
                path, 
                label_to_int, 
                fs=22050, 
                suffix='.wav', 
                audio_len=3, 
                f_paths=None):
        self.path = path
        self.label_to_int = label_to_int
        self.fs = fs
        self.suffix = suffix
        self.audio_len_samples = int(fs * audio_len)
        
        if f_paths is None:
            self.walk_path()
        else:
            self.files = [f for f in f_paths if f.endswith(self.suffix)]

    def walk_path(self):
        ''' yield all wav files contained in self.path
        '''
        root = os.path.expanduser(self.path)
        self.files = [os.path.join(dirpath, f) \
                      for dirpath, _, files in os.walk(root) \
                      for f in files if f.endswith(self.suffix)
                      ]

    def load_sol_item(self, idx):
        # get the label from the base directory's name
        file_path = self.files[idx]
        label = os.path.basename(os.path.dirname(file_path))

        waveform, fs = torchaudio.load(file_path) # load audio
    
        # resample the audio to the target sample rate
        resample = torchaudio.transforms.Resample(fs, self.fs)
        waveform = resample(waveform)

        return waveform[:, :self.audio_len_samples], label


    def __getitem__(self, idx):
        audio, label = self.load_sol_item(idx)

        sample = {
            "audio": audio, 
            "label": self.label_to_int[label]
        }
        return sample

    def __len__(self):
        return len(self.files)
