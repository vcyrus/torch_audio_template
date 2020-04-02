import librosa
import numpy as np

def mel_spectrogram(f_path, n_fft, win_length, hop_length, sr, n_mels, log_power=True, magnitude=True, input_fixed_length=None):
    audio, sr = librosa.load(f_path)
    audio = get_normalized_audio(audio)
    if input_fixed_length:
        audio = modify_file_length(audio, input_fixed_length)

    mel_spec = librosa.feature.melspectrogram(
          audio, 
          sr=sr, 
          n_mels=n_mels,
          hop_length=hop_length,
          n_fft=n_fft,
          win_length=win_length,
    )

    if magnitude:
        mel_spec = np.abs(mel_spec)

    if log_power:
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec.T # transpose to get T-F

def get_normalized_audio(y, head_room=0.005):
    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value

def modify_file_length(audio, input_fixed_length):
    if len(audio) < input_fixed_length:
        n_copies = int(np.ceil(input_fixed_length / len(audio)))
        audio_dup = np.tile(audio, (n_copies, 1))
        audio = audio_dup[:input_fixed_length]

    return audio

from torch.nn import MaxPool2d, Conv2d
import math

def calculate_output_dims(input_dims, layers, n_conv_channels):
    ''' Calculates the output dimensions given a tuple of tensor input dimensions
        and a list of {torch.nn.Conv2d} or {torch.nn.MaxPool2d} layers
        input dims: (batch_size, nc, w, h)
        layers: sequence of layers
    '''
    n_batch, n_channels, h, w = input_dims
    for l in layers:
        if isinstance(l, Conv2d):
            k_h, k_w = l.kernel_size
            p_h, p_w = l.padding
            s_h, s_w = l.stride
        if isinstance(l, MaxPool2d):
            if type(l.kernel_size) is not int:
                k_h, k_w = l.kernel_size
                s_h, s_w = l.stride 
            else:
                k_h = l.kernel_size
                k_w = k_h
                s_h = l.stride
                s_w = s_h
            p_h = 0
            p_w = 0
        h = math.floor((h + 2*p_h - k_h) / s_h + 1)
        w = math.floor((w + 2*p_w - k_w) / s_w + 1)

    return (n_batch, n_conv_channels, h, w)
