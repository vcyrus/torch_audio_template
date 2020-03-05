'''
Cyrus Vahidi

scripts/feature_extraction.py

Extract features from target_dir audio and output to output_dir 
'''


import numpy as np
import os

from argparse import ArgumentParser

from ..utils.feature_extractors import mel_spectrogram

def parse_args():
  parser = ArgumentParser(description="Extract mel spec features from audio target directory and dump to output dir in numpy")

  parser.add_argument("target_dir", type=str, help="path to input audio directory, absolute")
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
      default=2048,
      help="FFT window length"
  )
  parser.add_argument(
      "--hop_length",
      type=int,
      default=512,
      help="FFT hop length <= win_len"
  )
  parser.add_argument(
      "-sr",
      type=int,
      default=16000,
      help="Rate to resample audio at (default=16000)"
  )
  parser.add_argument(
      "--n_mels",
      type=int,
      default=96,
      help="Number of mel frequency bands to extract (default=96)"
  )

  return parser.parse_args()


def feature_extraction(target_dir, output_dir, feature, n_fft, win_length, 
                      hop_length, sr, n_mels):
    n_files = len([name for name in os.listdir(target_dir) if os.path.isfile(name)])
    idx = 1
    for entry in os.scandir(target_dir):
        f_path = entry.path
        if feature == "mel_spec":
            mel_spec = mel_spectrogram(
                f_path, n_fft, win_length, 
                hop_length, sr, n_mels, log_power=False)
        if feature == "log_mel_spec":
             mel_spec = mel_spectrogram(
                 f_path, n_fft, win_length, 
                 hop_length, sr, n_mels, log_power=True)

        path, _ = os.path.splitext(f_path)
        out_path = os.path.join(output_dir, os.path.join(path, '.data'), suffix='_mel')
        np.save(out_path, mel_spec)

        if os.exists(out_path):
            print("Extracted features {0} / {1}: {2}".format(idx, n_files, out_path))
        else:
            print("Failed {0} / {1}: {2}".format(idx, n_files, f_path))

        idx += 1

if __name__ == "__main__":
    args = parse_args()

    feature_extraction(
        args.target_dir,
        args.output_dir,
        args.feature,
        args.n_fft,
        args.win_length,
        args.hop_length,
        args.sr,
        args.n_mels
    )
