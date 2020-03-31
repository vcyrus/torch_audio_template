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
