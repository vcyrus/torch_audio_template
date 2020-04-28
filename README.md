# Torch Audio Boilerplate
A boilerplate project for deep learning for audio in PyTorch.

## Notes
- For now this is designed for personal and colleague use.

- Currently supports:
  - Loading mel spectrogram and CQT on the fly with nnAudio
  - Single class classification e.g SOL-DB instrument classification
  - Assumes that the dataset is provided as raw audio files, where subdirectories in the path to the dataset are grouped by label and named as the containing files' label

## Installation
- `git clone`
- `cd torch_audio_template/`
- `pip install requirements.txt`

Example use:
- `python -m src.scripts.train SOL_DIR --audio_len=1 --batch_size=64 -lr=0.001 --dropout_rate=0.25 --n_epochs=5 --weight_decay=0.00001 -transform='cqt' --n_bins=84`

- `SOL_DIR/`
  - `Alto-Sax  Bassoon  Cello  Clarinet  Flute  Guitar  Harp  Horn  Oboe  Trombone  Trumpet  Violin`
