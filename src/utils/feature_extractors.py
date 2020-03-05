import librosa
import numpy as np

def mel_spectrogram(file, n_fft, win_length, hop_length, sr, n_mels, log_power=True, magnitude=True):
    audio, sr = librosa.load(file.path)
    audio = get_normalized_audio(audio)

    mel_spec = librosa.feature.mel_spectrogram(
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

    return mel_spec

def get_normalized_audio(y, head_room=0.005):
    mean_value = np.mean(y)
    y -= mean_value

    max_value = max(abs(y)) + head_room
    return y / max_value





