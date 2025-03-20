import os
import librosa
import numpy as np

input_dir = "data/preprocessed/"  
output_dir = "data/mel_spectrograms/" 
os.makedirs(output_dir, exist_ok=True)

TARGET_SAMPLE_RATE = 16000  
N_MELS = 128  
HOP_LENGTH = 512 
N_FFT = 2048 

def create_mel_spectrogram(wav_file):
    try:
        audio, sr = librosa.load(wav_file, sr=TARGET_SAMPLE_RATE, mono=True)

        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=TARGET_SAMPLE_RATE, 
                                                         n_mels=N_MELS, n_fft=N_FFT, 
                                                         hop_length=HOP_LENGTH)

        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        output_file = os.path.join(output_dir, os.path.basename(wav_file).replace(".wav", ".npy"))
        np.save(output_file, mel_spectrogram_db)
        print(f"Processed and saved Mel spectrogram for {wav_file} -> {output_file}")
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")

wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".wav")]

for wav in wav_files:
    create_mel_spectrogram(wav)
