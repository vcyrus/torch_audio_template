'''
Cyrus Vahidi

src/scripts/feature_extraction.py

Extract features and labels from audio located in target_dir and output to output_dir 
Default features are log mel spectrograms extracted with Librosa.

This example scripts is for a binary classification task.
The csv is structured with file name (no extension) in column 'itemid'
and label in column 'hasbird'.
Loads audio files and saves mel spec features
'''

import numpy as np
import pandas as pd
import os

from argparse import ArgumentParser

from src.utils.feature_extractors import mel_spectrogram
from src.utils.files import save_array

def parse_args():
  parser = ArgumentParser(description="Extract mel spec features from audio target directory and dump to output dir in numpy")

  parser.add_argument("target_dir", type=str, help="path to input audio directory, absolute")
  parser.add_argument("csv_path", type=str, help="filenames and labels csv")
  parser.add_argument("output_dir", type=str, help="path to write features to, absolute")
  parser.add_argument("-feature", type=str, default="log_mel_spec", help="feature to extract e.g mel_spec, log_mel_spec")
  parser.add_argument(
      "--n_fft", 
      type=int, 
      default=2048, 
      help="FFT size, default=2048"
  )
  parser.add_argument(
      "--win_length",
      type=int,
      default=1024,
      help="FFT window length"
  )
  parser.add_argument(
      "--hop_length",
      type=int,
      default=1024,
      help="FFT hop length <= win_len"
  )
  parser.add_argument(
      "-sr",
      type=int,
      default=44100,
      help="Rate to resample audio at (default=16000)"
  )
  parser.add_argument(
      "--n_mels",
      type=int,
      default=96,
      help="Number of mel frequency bands to extract (default=96)"
  )

  return parser.parse_args()


def feature_extraction(target_dir, csv, output_dir, feature, n_fft, win_length, hop_length, sr, n_mels):
    n_files = len([name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, name))])
    df_csv = pd.read_csv(csv)

    idx = 1
    for entry in os.scandir(target_dir):
        f_path = entry.path
        log_power = feature == "log_mel_spec"
        mel_spec = mel_spectrogram(
                f_path, n_fft, win_length, 
                hop_length, sr, n_mels, log_power=log_power
        )

        f_name = entry.name   
        f_base, f_ext = os.path.splitext(f_name)
        out_path = os.path.join(
            output_dir, 
            f_name.replace(f_ext, '.npy')
        )
        save_array(mel_spec, out_path, suffix='_mel')

        # get label matching filename from the csv
        label = df_csv.loc[df_csv['itemid'].str.startswith(f_base)]['hasbird'].values[0]
        save_array(np.array([label], dtype=float), out_path, suffix='_label')

        if os.path.exists(out_path.replace('.npy', '_mel.npy')) and \
            os.path.exists(out_path.replace('.npy', '_label.npy')):
            print("Extracted features {0} / {1}: {2}".format(idx, n_files, out_path))
        else:
            print("Failed {0} / {1}: {2}".format(idx, n_files, f_path))
        idx += 1

if __name__ == "__main__":
    args = parse_args()

    feature_extraction(
        args.target_dir,
        args.csv_path,
        args.output_dir,
        args.feature,
        args.n_fft,
        args.win_length,
        args.hop_length,
        args.sr,
        args.n_mels
    )
